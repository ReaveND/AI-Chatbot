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
DATASET_PATH = "dataset.txt"
MODEL_DIR = "saved_model"
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.3
EPOCHS = 500
BATCH_SIZE = 16
LEARNING_RATE = 0.001
TEACHER_FORCING_START = 0.7
TEACHER_FORCING_END = 0.3
CLIP_GRAD = 1.0
VAL_SPLIT = 0.15
EARLY_STOPPING_PATIENCE = 60
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset helpers ────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def sentence_to_indexes(sentence: str, vocab: Vocabulary):
    indexes = vocab.sentence_to_indexes(sentence)
    indexes.append(Vocabulary.EOS_token)
    return indexes


def create_batches(pairs, vocab, batch_size, shuffle=True):
    indexed_pairs = [
        (sentence_to_indexes(inp, vocab), sentence_to_indexes(out, vocab))
        for inp, out in pairs
    ]

    if shuffle:
        random.shuffle(indexed_pairs)

    for i in range(0, len(indexed_pairs), batch_size):
        chunk = indexed_pairs[i:i + batch_size]
        src_sequences = [p[0] for p in chunk]
        tgt_sequences = [[Vocabulary.SOS_token] + p[1] for p in chunk]

        src_max_len = max(len(seq) for seq in src_sequences)
        tgt_max_len = max(len(seq) for seq in tgt_sequences)

        src_batch = [seq + [Vocabulary.PAD_token] * (src_max_len - len(seq)) for seq in src_sequences]
        tgt_batch = [seq + [Vocabulary.PAD_token] * (tgt_max_len - len(seq)) for seq in tgt_sequences]

        src_tensor = torch.tensor(src_batch, dtype=torch.long, device=DEVICE)
        tgt_tensor = torch.tensor(tgt_batch, dtype=torch.long, device=DEVICE)

        yield src_tensor, tgt_tensor


def split_dataset(pairs):
    random.shuffle(pairs)
    val_size = max(1, int(len(pairs) * VAL_SPLIT))
    val_pairs = pairs[:val_size]
    train_pairs = pairs[val_size:]
    return train_pairs, val_pairs


# ── Training/eval loops ────────────────────────────────────────────────────────

def run_epoch(model, pairs, vocab, optimizer, criterion, teacher_forcing_ratio, train_mode=True):
    total_loss = 0.0
    total_batches = 0

    if train_mode:
        model.train()
    else:
        model.eval()

    with torch.set_grad_enabled(train_mode):
        for src_batch, tgt_batch in create_batches(
            pairs, vocab, BATCH_SIZE, shuffle=train_mode
        ):
            if train_mode:
                optimizer.zero_grad()

            output = model(src_batch, tgt_batch, teacher_forcing_ratio)
            vocab_size = output.shape[-1]

            output_flat = output[:, 1:, :].reshape(-1, vocab_size)
            target_flat = tgt_batch[:, 1:].reshape(-1)

            loss = criterion(output_flat, target_flat)

            if train_mode:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
                optimizer.step()

            total_loss += loss.item()
            total_batches += 1

    return total_loss / max(total_batches, 1)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    print(f"Using device: {DEVICE}")

    pairs = load_dataset(DATASET_PATH)
    print(f"Loaded {len(pairs)} conversation pairs")

    if len(pairs) < 4:
        raise ValueError("Dataset is too small. Add more conversation pairs before training.")

    vocab = build_vocab(pairs)
    train_pairs, val_pairs = split_dataset(pairs)

    print(f"Vocabulary size: {vocab.n_words} tokens")
    print(f"Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)}")

    encoder = Encoder(vocab.n_words, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
    decoder = Decoder(vocab.n_words, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=Vocabulary.PAD_token, label_smoothing=0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=20
    )

    best_train_loss = float("inf")
    epochs_without_improvement = 0
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"\nTraining for up to {EPOCHS} epochs...\n")

    for epoch in range(1, EPOCHS + 1):
        progress = (epoch - 1) / max(EPOCHS - 1, 1)
        teacher_forcing_ratio = TEACHER_FORCING_START - (
            (TEACHER_FORCING_START - TEACHER_FORCING_END) * progress
        )

        train_loss = run_epoch(
            model,
            train_pairs,
            vocab,
            optimizer,
            criterion,
            teacher_forcing_ratio,
            train_mode=True,
        )
        val_loss = run_epoch(
            model,
            val_pairs,
            vocab,
            optimizer,
            criterion,
            teacher_forcing_ratio=0.0,
            train_mode=False,
        )
        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:4d}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"TF: {teacher_forcing_ratio:.3f} | LR: {lr_now:.6f}"
            )

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            epochs_without_improvement = 0
            torch.save(encoder.state_dict(), os.path.join(MODEL_DIR, "encoder.pt"))
            torch.save(decoder.state_dict(), os.path.join(MODEL_DIR, "decoder.pt"))
            vocab.save(os.path.join(MODEL_DIR, "vocab.pkl"))
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping triggered at epoch {epoch} "
                f"after {EARLY_STOPPING_PATIENCE} epochs without improvement."
            )
            break

    print(f"\nTraining complete! Best train loss: {best_train_loss:.4f}")
    print(f"Model saved to '{MODEL_DIR}/'")


if __name__ == "__main__":
    main()
