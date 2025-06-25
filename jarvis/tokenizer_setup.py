from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from typing import Optional
import os

SPECIAL_TOKENS = ["<PAD>", "<EOS>", "<SEP>", "<UNK>"]

def train_tokenizer(text_file_path: str, vocab_size: int = 2048, save_path: str = "models/tokenizerv2.json", add_if_new=False):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

    SPECIAL_TOKENS = ["<PAD>", "<EOS>", "<SEP>", "<UNK>"]

    if add_if_new and os.path.exists(save_path):
        # Load existing tokenizer
        tokenizer = Tokenizer.from_file(save_path)
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()

        # Extract old vocab
        old_vocab = tokenizer.get_vocab()
        old_tokens = set(old_vocab.keys())

        # Read new text and extract new tokens
        with open(text_file_path, "r", encoding="utf-8") as f:
            new_lines = f.readlines()

        temp_tokenizer = Tokenizer(models.BPE())
        temp_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=2,
            special_tokens=SPECIAL_TOKENS,
        )
        temp_tokenizer.train_from_iterator(new_lines, trainer=trainer)
        new_vocab = temp_tokenizer.get_vocab()

        new_tokens = [tok for tok in new_vocab if tok not in old_tokens]

        print(f"🔍 Found {len(new_tokens)} new tokens.")

        # Add only new tokens
        tokenizer.add_tokens(new_tokens)

    else:
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=2,
            special_tokens=SPECIAL_TOKENS
        )
        tokenizer.train([text_file_path], trainer=trainer)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.add_special_tokens(SPECIAL_TOKENS)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)

    print(f"✅ Tokenizer saved to {save_path}")
    print(f"📏 Vocab size: {tokenizer.get_vocab_size()}")
    for token in SPECIAL_TOKENS:
        print(f"🔖 {token} → ID {tokenizer.token_to_id(token)}")

    return tokenizer


def load_tokenizer(path: str = "models/tokenizerv2.json") -> Tokenizer:
    """
    Loads a saved BPE tokenizer and resets its pre-tokenizer/decoder for reliability.
    """
    tokenizer = Tokenizer.from_file(path)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer
