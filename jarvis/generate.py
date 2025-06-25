import torch
import torch.nn.functional as F

def generate_text(model, tokenizer, prompt, max_length=50, eos_token_id=None, temperature=1.0, top_k=0, device='cpu'):
    model.eval()
    input_ids = tokenizer.encode(prompt).ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    for _ in range(max_length):
        with torch.no_grad():
            logits = model(input_tensor)
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(logits, top_k)
                probs = torch.zeros_like(logits).scatter_(1, top_k_indices, F.softmax(top_k_values, dim=-1))
            else:
                probs = F.softmax(logits, dim=-1)

            next_token_id = torch.multinomial(probs, num_samples=1)
            input_tensor = torch.cat([input_tensor, next_token_id], dim=1)

            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break

    output_ids = input_tensor[0].tolist()
    return tokenizer.decode(output_ids)
