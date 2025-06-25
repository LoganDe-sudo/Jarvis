
# 🧠 Jarvis – Dynamic Vocabulary Transformer

**Jarvis** is a GPT-style transformer language model that **supports dynamic vocabulary expansion** — enabling the model to learn new words/tokens *on the fly* without restarting training or breaking core model behavior. It's ideal for **AI assistants that grow smarter over time** with custom data.

---

## 🚀 Features

- ✅ **Dynamic Embedding + Head**: `ExpandableEmbedding` & `ExpandableLinearHead` grow without reinitializing
- ✅ **Safe vocabulary expansion**: new tokens can be added without resetting training
- ⚠️ **Flexible model loading**: existing weights are reused, and only mismatched vocab layers are skipped
- ✅ **No training loss on expansion**: continue training even after tokenizer grows
- ✅ Integrated with HuggingFace-compatible tokenizer (`tokenizers` lib)
- ✅ Supports gradient clipping, AMP training, early stopping
- 📉 Tracks loss, perplexity during training

---

## 🏗️ Architecture

- Transformer encoder stack (custom-built from scratch)
- `MultiHeadAttention` using PyTorch's native implementation
- Sinusoidal positional encoding
- Causal (auto-regressive) masking
- Modular layer design for easy scaling (layers, dimensions, vocab)

---

## 📦 Project Structure

```
jarvis/
├── model.py             # Core Transformer + dynamic vocab model
├── tokenizer_setup.py   # BPE Tokenizer training and updates
├── train.py             # Continual training pipeline
├── generate.py          # Inference script for text generation
├── testchat.py          # Interactive testing script
```

---

## 🏁 Quick Start

1. **Install dependencies**
```bash
pip install torch tokenizers scikit-learn tqdm
```

2. **Prepare your data**  
In `.jsonl` format:
```json
{"input": "What's your name?", "response": "I'm Jarvis."}
```

3. **Train the model**
```bash
python train.py
```

4. **Generate output**
```bash
python generate.py
```

---

## 🔁 Continual Learning Flow

This is how Jarvis evolves:

```
[train on dataset A] → [add new data] → [tokenizer grows] → [model.expand_vocab()] → [resume training] → [repeat forever]
```

Even with new tokens, Jarvis doesn’t lose old knowledge.

---

## 🧪 What Makes This Special?

Normally, changing the tokenizer breaks the model:

- ❌ You'd need to **start training from scratch** or
- ❌ You’d get **state_dict loading errors** due to mismatched sizes

**But Jarvis fixes this:**

- ✨ `ExpandableEmbedding`: expands to fit new tokens, keeps old weights
- ✨ `ExpandableLinearHead`: maps expanded embeddings to new vocab
- ✨ `load_flexible_state_dict()`: loads all compatible weights, safely skips mismatches

🧠 *"Partial mismatch? No problem — the brain remains intact."*

---

## ✅ Example Result

- You train on Set B
- Then Set A
- Then again Set B  
→ Perplexity **starts low again** — proving Jarvis **remembers past datasets**, even with vocab growth.

---

## ⚠️ Note on Model Loading

Jarvis **does flexible (not 100% identical) loading**:
- It reuses *all matching parameters*
- It **skips** old embedding/head weights if vocab size changed
- But this is *intentional* to preserve training

---

## 🏭 Scale & Future Vision

Jarvis is **modular** — you can:

- 🔧 Increase model depth, attention heads, vocab size
- 🧠 Train it on larger datasets
- 🔌 Plug it into tools (voice input, web search, APIs)

Big companies with large compute can adopt this **as a lifelong AI learner framework**.

---

## 📣 Credits

Crafted with care by **Logesh (a.k.a. Jarvis Creator)**  
> "Let your model grow smarter like a real brain — forever." 💡
