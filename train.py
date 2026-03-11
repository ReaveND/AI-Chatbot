"""
train.py  —  Train the Seq2Seq LSTM chatbot.

Usage:
    python train.py

Saves trained weights and vocabulary to saved_model/
"""

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim

from vocabulary import Vocabulary, preprocess
from model import Encoder, Decoder, Seq2Seq

# ── Hyperparameters ────────────────────────────────────────────────────────────
DATASET_PATH          = "dataset.txt"
MODEL_DIR             = "saved_model"
HIDDEN_SIZE           = 256
NUM_LAYERS            = 2
DROPOUT               = 0.3
EPOCHS                = 500
LEARNING_RATE         = 0.001
TEACHER_FORCING_RATIO = 0.5
CLIP_GRAD             = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset helpers ────────────────────────────────────────────────────────────

def load_dataset(path: str):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "\t" not in line:
                continue
            inp, out = line.split("\t", 1)
            inp, out = preprocess(inp), preprocess(out)
            if inp and out:
                pairs.append((inp, out))
    return pairs


def build_vocab(pairs):
    vocab = Vocabulary()
    for inp, out in pairs:
        vocab.add_sentence(inp)
        vocab.add_sentence(out)
    return vocab


def sentence_to_tensor(sentence: str, vocab: Vocabulary, device):
    """Convert a sentence string to a 1-D tensor ending with <EOS>."""
    indexes = vocab.sentence_to_indexes(sentence)
    indexes.append(Vocabulary.EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device)


# ── Training loop ──────────────────────────────────────────────────────────────

def train_epoch(model, pairs, vocab, optimizer, criterion):
    model.train()
    total_loss = 0.0
    random.shuffle(pairs)

    sos = torch.tensor([Vocabulary.SOS_token], device=DEVICE)

    for pair in pairs:
        inp_t = sentence_to_tensor(pair[0], vocab, DEVICE)   # (src_len,)
        tgt_t = sentence_to_tensor(pair[1], vocab, DEVICE)   # (trg_len,)

        # Add <SOS> prefix to target and batch-dim to both
        tgt_with_sos = torch.cat([sos, tgt_t]).unsqueeze(0)  # (1, trg_len+1)
        inp_t = inp_t.unsqueeze(0)                           # (1, src_len)

        optimizer.zero_grad()

        output = model(inp_t, tgt_with_sos, TEACHER_FORCING_RATIO)
        # output: (1, trg_len+1, vocab_size)

        vocab_size = output.shape[-1]
        # Skip timestep-0 (SOS position) in predictions; compare to target[1:]
        output_flat = output[:, 1:, :].reshape(-1, vocab_size)  # (trg_len, V)
        target_flat = tgt_with_sos[:, 1:].reshape(-1)           # (trg_len,)

        loss = criterion(output_flat, target_flat)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(pairs)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Using device: {DEVICE}")

    pairs = load_dataset(DATASET_PATH)
    print(f"Loaded {len(pairs)} conversation pairs")

    vocab = build_vocab(pairs)
    print(f"Vocabulary size: {vocab.n_words} tokens")

    encoder = Encoder(vocab.n_words, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
    decoder = Decoder(vocab.n_words, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
    model   = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=Vocabulary.PAD_token)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=30, verbose=False
    )

    best_loss = float("inf")
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"\nTraining for {EPOCHS} epochs...\n")

    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_epoch(model, pairs, vocab, optimizer, criterion)
        scheduler.step(avg_loss)

        if epoch % 50 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:4d}/{EPOCHS}  |  Loss: {avg_loss:.4f}  |  LR: {lr_now:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(encoder.state_dict(), os.path.join(MODEL_DIR, "encoder.pt"))
            torch.save(decoder.state_dict(), os.path.join(MODEL_DIR, "decoder.pt"))
            vocab.save(os.path.join(MODEL_DIR, "vocab.pkl"))

    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    print(f"Model saved to '{MODEL_DIR}/'")


if __name__ == "__main__":
    main()
