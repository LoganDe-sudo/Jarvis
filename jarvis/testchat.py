import torch
from tokenizer_setup import load_tokenizer
from model import Jarvis
from generate import generate_text # Uses your current canvas implementation

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Tokenizer ---
tokenizer = load_tokenizer()
vocab_size = tokenizer.get_vocab_size()
PAD_TOKEN_ID = tokenizer.token_to_id("<PAD>")
EOS_TOKEN_ID = tokenizer.token_to_id("<EOS>")

# --- Load Model ---
config = {
    "embed_dim": 128,
    "num_heads": 4,
    "ff_hidden": 512,
    "max_len": 512,
    "num_layers": 4,
    "dropout": 0.1
}

model = Jarvis(vocab_size, config)
model.load_state_dict(torch.load("models/jarvis_model_v2.pt", map_location=device))
model.to(device)
model.eval()

print("💬 Jarvis is ready! Type something (or type 'exit' to quit):")

# --- Chat Loop ---
while True:
    prompt = input("\nAsk me anything!: ").strip()
    if prompt.lower() in {"exit", "quit"}:
        print("👋 Goodbye!")
        break

    # Optional: dynamic parameters (or you can hardcode)
    temperature = 1.0
    top_k = 20

    response = generate_text(
        model, tokenizer, prompt,
        max_length=100,
        eos_token_id=EOS_TOKEN_ID,
        temperature=temperature,
        top_k=top_k,
        device=device
    )

    print(response)

