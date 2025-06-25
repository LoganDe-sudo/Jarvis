# Updated train.py to support dynamic tokenizer + dynamic model
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tokenizer_setup import train_tokenizer, load_tokenizer
from model import Jarvis
from pathlib import Path
import os
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler

# --- Configuration ---
CONFIG = {
    "data_path": "data/hi.jsonl",
    "tmp_text_path": "data/tmp_training.txt",
    "model_save_dir": "models",
    "model_filename": "jarvis_model_v2.pt",
    "tokenizer_filename": "tokenizerv2.json",
    "learning_rate": 3e-4,
    "num_epochs": 50,
    "batch_size": 16,
    "grad_accum_steps": 1,
    "weight_decay": 0.01,
    "clip_grad_norm": 1.0,
    "patience": 5,
    "min_delta": 0.001,
    "train_max_length": 128,
    "vocab_size": 2048,
    "start_fresh": False
}

# --- Paths ---
Path(CONFIG["model_save_dir"]).mkdir(parents=True, exist_ok=True)
MODEL_PATH = os.path.join(CONFIG["model_save_dir"], CONFIG["model_filename"])
TOKENIZER_PATH = os.path.join(CONFIG["model_save_dir"], CONFIG["tokenizer_filename"])

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Data formatting ---
ACCEPTABLE_INPUT_KEYS = ["input"] # Add more keys as needed
ACCEPTABLE_RESPONSE_KEYS = ["response"] # Add more keys as needed

def find_key(d, candidates):
    return next((key for key in candidates if key in d), None)

def load_flexible_state_dict(model, checkpoint):
    model_state = model.state_dict()
    loaded_state = {}

    for name, param in checkpoint.items():
        if name not in model_state:
            print(f"⏭️ Skipping {name}: not in model.")
            continue

        model_param = model_state[name]
        if param.shape == model_param.shape:
            loaded_state[name] = param
        else:
            print(f"🧩 Expanding layer: {name} | checkpoint: {param.shape} -> model: {model_param.shape}")
            temp = model_param.clone()
            slices = tuple(slice(0, min(a, b)) for a, b in zip(param.shape, model_param.shape))
            temp[slices] = param[slices]
            loaded_state[name] = temp

    model.load_state_dict(loaded_state, strict=False)


raw_lines = []
with open(CONFIG["data_path"], "r", encoding="utf-8") as f:
    for line in f:
        example = json.loads(line)
        input_key = find_key(example, ACCEPTABLE_INPUT_KEYS)
        response_key = find_key(example, ACCEPTABLE_RESPONSE_KEYS)
        if not input_key or not response_key:
            continue
        user_input = example[input_key].strip()
        bot_reply = example[response_key].strip()
        raw_lines.append(f"{user_input} <SEP> {bot_reply} <EOS>")

with open(CONFIG["tmp_text_path"], "w", encoding="utf-8") as f:
    f.write("\n".join(raw_lines))

# --- Tokenizer ---
tokenizer = None
try:
    tokenizer = train_tokenizer(
        CONFIG["tmp_text_path"],
        vocab_size=CONFIG["vocab_size"],
        save_path=TOKENIZER_PATH,
        add_if_new=True
    )
except Exception as e:
    print(f"⚠️ Could not load existing tokenizer. Reason: {e}")
    print("🔁 Re-training tokenizer from scratch.")
    tokenizer = train_tokenizer(
        CONFIG["tmp_text_path"],
        vocab_size=CONFIG["vocab_size"],
        save_path=TOKENIZER_PATH,
        add_if_new=False
    )


vocab_size = tokenizer.get_vocab_size()
PAD_TOKEN_ID = tokenizer.token_to_id("<PAD>")
EOS_TOKEN_ID = tokenizer.token_to_id("<EOS>")

# --- Dataset ---
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512, pad_token_id=None):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer.encode(self.texts[idx])
        ids = encoded.ids[:self.max_length]
        padding = [self.pad_token_id] * (self.max_length - len(ids))
        ids += padding
        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        labels = torch.tensor(ids[1:], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}

train_texts, val_texts = train_test_split(raw_lines, test_size=0.1, random_state=42)
train_dataset = TextDataset(train_texts, tokenizer, CONFIG["train_max_length"], PAD_TOKEN_ID)
val_dataset = TextDataset(val_texts, tokenizer, CONFIG["train_max_length"], PAD_TOKEN_ID)
train_loader = DataLoader(train_dataset, CONFIG["batch_size"], shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, CONFIG["batch_size"], shuffle=False, drop_last=False)

# --- Model Init ---
model_config = {
    "embed_dim": 128,
    "num_heads": 4,
    "ff_hidden": 512,
    "max_len": 512,
    "num_layers": 4,
    "dropout": 0.1
}
model = Jarvis(vocab_size, model_config).to(device)

# No partial loading — only match vocab if starting from scratch
if not CONFIG["start_fresh"] and os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    load_flexible_state_dict(model, checkpoint)
    print(f"✅ Loaded model from: {MODEL_PATH}")
else:
    print("🆕 No checkpoint loaded — starting from scratch.")

# Expand embedding/head if vocab changed
model.expand_vocab(vocab_size)

# --- Optim, Loss, Sched ---
optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
scaler = GradScaler() if device.type == 'cuda' else None

# --- Early Stopping ---
class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_model = None

    def check(self, loss, model):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.best_model = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

early_stopper = EarlyStopping(CONFIG["patience"], CONFIG["min_delta"])

# --- Training ---
print("\n🚀 Starting training...")
for epoch in range(CONFIG["num_epochs"]):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}", leave=False):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with autocast(enabled=scaler is not None):
            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, model.get_vocab_size()), labels.view(-1))

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["clip_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["clip_grad_norm"])
            optimizer.step()

        optimizer.zero_grad()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, model.get_vocab_size()), labels.view(-1))
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    scheduler.step(avg_val_loss)

    print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Perplexity: {math.exp(avg_loss):.2f}")

    if early_stopper.check(avg_val_loss, model):
        print(f"🛑 Early stopping at epoch {epoch+1}")
        break

# --- Save ---
final_model = early_stopper.best_model or model.state_dict()
torch.save(final_model, MODEL_PATH)
print(f"💾 Model saved to {MODEL_PATH}")
