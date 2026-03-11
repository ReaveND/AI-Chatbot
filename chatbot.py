"""
chatbot.py  —  Inference engine for the trained Seq2Seq chatbot.
"""

import math
import os
from collections import Counter

import torch

from vocabulary import Vocabulary, preprocess
from model import Encoder, Decoder

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_DIR = "saved_model"
DATASET_PATH = "dataset.txt"
HIDDEN_SIZE = 256
NUM_LAYERS = 2
MAX_RESPONSE_LEN = 24
MIN_RESPONSE_LEN = 3
BEAM_WIDTH = 5
LENGTH_PENALTY_ALPHA = 0.75
REPETITION_PENALTY = 1.2
BM25_K1 = 1.5
BM25_B = 0.75
RETRIEVAL_CONFIDENCE_THRESHOLD = 3.0
MAX_UNKNOWN_RATIO = 0.45

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Chatbot:
    def __init__(self):
        vocab_path = os.path.join(MODEL_DIR, "vocab.pkl")
        encoder_path = os.path.join(MODEL_DIR, "encoder.pt")
        decoder_path = os.path.join(MODEL_DIR, "decoder.pt")

        self.qa_pairs = self._load_qa_pairs(DATASET_PATH)
        self._bm25_index = self._build_bm25_index(self.qa_pairs)

        self.vocab = None
        self.encoder = None
        self.decoder = None

        model_files_present = all(os.path.exists(p) for p in [vocab_path, encoder_path, decoder_path])
        if not model_files_present:
            if not self.qa_pairs:
                raise FileNotFoundError(
                    "Neither trained model nor dataset was found. Add dataset.txt or run 'python train.py'."
                )
            return

        self.vocab = Vocabulary.load(vocab_path)
        self.encoder = Encoder(self.vocab.n_words, HIDDEN_SIZE, NUM_LAYERS, dropout=0.0).to(DEVICE)
        self.decoder = Decoder(self.vocab.n_words, HIDDEN_SIZE, NUM_LAYERS, dropout=0.0).to(DEVICE)

        try:
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=DEVICE))
            self.decoder.load_state_dict(torch.load(decoder_path, map_location=DEVICE))
        except RuntimeError:
            # Model architecture changed: keep retrieval available until retraining.
            self.encoder = None
            self.decoder = None
            self.vocab = None
            return

        self.encoder.eval()
        self.decoder.eval()

    def _load_qa_pairs(self, path):
        if not os.path.exists(path):
            return []

        pairs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "\t" not in line:
                    continue
                user_q, bot_a = line.split("\t", 1)
                user_q = preprocess(user_q)
                bot_a = bot_a.strip()
                if user_q and bot_a:
                    pairs.append((user_q, bot_a))
        return pairs

    def _build_bm25_index(self, pairs):
        if not pairs:
            return None

        doc_tokens = [q.split() for q, _ in pairs]
        doc_freq = Counter()
        term_freqs = []
        doc_lengths = []

        for tokens in doc_tokens:
            token_counts = Counter(tokens)
            term_freqs.append(token_counts)
            doc_lengths.append(len(tokens))
            doc_freq.update(token_counts.keys())

        num_docs = len(doc_tokens)
        avg_doc_len = sum(doc_lengths) / max(num_docs, 1)
        idf = {
            term: math.log(1 + (num_docs - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

        return {
            "term_freqs": term_freqs,
            "doc_lengths": doc_lengths,
            "avg_doc_len": avg_doc_len,
            "idf": idf,
        }

    def _retrieve_response(self, cleaned_query: str):
        if not self._bm25_index:
            return None, float("-inf")

        query_terms = cleaned_query.split()
        idf = self._bm25_index["idf"]
        term_freqs = self._bm25_index["term_freqs"]
        doc_lengths = self._bm25_index["doc_lengths"]
        avg_doc_len = self._bm25_index["avg_doc_len"]

        best_score = float("-inf")
        best_response = None

        for idx, tf in enumerate(term_freqs):
            score = 0.0
            doc_len = doc_lengths[idx]

            for term in query_terms:
                if term not in tf:
                    continue

                term_idf = idf.get(term, 0.0)
                freq = tf[term]
                numerator = freq * (BM25_K1 + 1)
                denominator = freq + BM25_K1 * (1 - BM25_B + BM25_B * (doc_len / max(avg_doc_len, 1e-6)))
                score += term_idf * (numerator / denominator)

            if score > best_score:
                best_score = score
                best_response = self.qa_pairs[idx][1]

        return best_response, best_score

    def _decode_with_beam_search(self, encoder_outputs, hidden, src_mask):
        beams = [([Vocabulary.SOS_token], hidden, 0.0, set())]
        completed = []

        for _ in range(MAX_RESPONSE_LEN):
            expanded = []
            for tokens, beam_hidden, score, used_tokens in beams:
                last_token = tokens[-1]
                if last_token == Vocabulary.EOS_token and len(tokens) > 1:
                    completed.append((tokens, score))
                    continue

                dec_input = torch.tensor([last_token], device=DEVICE)
                output, next_hidden = self.decoder(
                    dec_input,
                    beam_hidden,
                    encoder_outputs,
                    src_mask,
                )
                logits = output.squeeze(0)

                for token_id in used_tokens:
                    logits[token_id] /= REPETITION_PENALTY

                log_probs = torch.log_softmax(logits, dim=-1)
                top_values, top_indices = torch.topk(log_probs, k=min(BEAM_WIDTH, log_probs.numel()))

                for i in range(top_indices.numel()):
                    next_token = top_indices[i].item()
                    next_score = score + top_values[i].item()
                    next_tokens = tokens + [next_token]
                    next_used = set(used_tokens)
                    if next_token not in (Vocabulary.SOS_token, Vocabulary.EOS_token, Vocabulary.PAD_token):
                        next_used.add(next_token)
                    expanded.append((next_tokens, next_hidden, next_score, next_used))

            if not expanded:
                break

            expanded.sort(
                key=lambda item: item[2] / ((len(item[0]) ** LENGTH_PENALTY_ALPHA) or 1.0),
                reverse=True,
            )
            beams = expanded[:BEAM_WIDTH]

            if all(tokens[-1] == Vocabulary.EOS_token for tokens, *_ in beams):
                for tokens, _, score, _ in beams:
                    completed.append((tokens, score))
                break

        candidates = completed if completed else [(tokens, score) for tokens, _, score, _ in beams]
        candidates.sort(
            key=lambda item: item[1] / ((len(item[0]) ** LENGTH_PENALTY_ALPHA) or 1.0),
            reverse=True,
        )
        return candidates[0][0]

    def respond(self, sentence: str) -> str:
        cleaned = preprocess(sentence)
        if not cleaned:
            return "Could you rephrase that? I didn't quite catch it."

        retrieval_response, retrieval_score = self._retrieve_response(cleaned)

        if self.vocab is not None:
            query_tokens = cleaned.split()
            unk_count = sum(1 for token in query_tokens if token not in self.vocab.word2index)
            unk_ratio = unk_count / max(len(query_tokens), 1)
            if retrieval_response and (retrieval_score >= RETRIEVAL_CONFIDENCE_THRESHOLD or unk_ratio > MAX_UNKNOWN_RATIO):
                return retrieval_response
        elif retrieval_response:
            return retrieval_response

        if self.encoder is None or self.decoder is None or self.vocab is None:
            return "I can help with admissions, attendance, fees, exams, and timetable queries."

        indexes = self.vocab.sentence_to_indexes(cleaned)
        indexes.append(Vocabulary.EOS_token)
        input_tensor = torch.tensor(indexes, dtype=torch.long, device=DEVICE).unsqueeze(0)

        with torch.no_grad():
            src_mask = input_tensor.ne(Vocabulary.PAD_token)
            encoder_outputs, hidden = self.encoder(input_tensor)
            token_sequence = self._decode_with_beam_search(encoder_outputs, hidden, src_mask)

        response_words = []
        for token_id in token_sequence[1:]:
            if token_id == Vocabulary.EOS_token:
                break
            word = self.vocab.index2word.get(token_id, "")
            if word and word not in ("<PAD>", "<SOS>", "<EOS>", "<UNK>"):
                response_words.append(word)

        if len(response_words) < MIN_RESPONSE_LEN and retrieval_response:
            return retrieval_response

        if len(response_words) < MIN_RESPONSE_LEN:
            return "I can help with admissions, attendance, fees, exams, and timetable queries."

        return " ".join(response_words)


if __name__ == "__main__":
    print("Loading chatbot...")
    bot = Chatbot()
    print("Chatbot ready! Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "bye"):
            print("Bot: Goodbye! Have a great day!")
            break
        if user_input:
            print(f"Bot: {bot.respond(user_input)}")
