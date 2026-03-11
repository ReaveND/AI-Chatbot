"""
chatbot.py  —  Inference engine for the trained Seq2Seq chatbot.
"""

import os
import torch

from vocabulary import Vocabulary, preprocess
from model import Encoder, Decoder

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_DIR = "saved_model"
HIDDEN_SIZE = 256
NUM_LAYERS = 2
MAX_RESPONSE_LEN = 20
MIN_RESPONSE_LEN = 2
BEAM_WIDTH = 3
LENGTH_PENALTY_ALPHA = 0.7
REPETITION_PENALTY = 1.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Chatbot:
    def __init__(self):
        vocab_path = os.path.join(MODEL_DIR, "vocab.pkl")
        encoder_path = os.path.join(MODEL_DIR, "encoder.pt")
        decoder_path = os.path.join(MODEL_DIR, "decoder.pt")

        if not all(os.path.exists(p) for p in [vocab_path, encoder_path, decoder_path]):
            raise FileNotFoundError(
                "Trained model not found. Run 'python train.py' first."
            )

        self.vocab = Vocabulary.load(vocab_path)
        self.encoder = Encoder(self.vocab.n_words, HIDDEN_SIZE, NUM_LAYERS, dropout=0.0).to(DEVICE)
        self.decoder = Decoder(self.vocab.n_words, HIDDEN_SIZE, NUM_LAYERS, dropout=0.0).to(DEVICE)

        self.encoder.load_state_dict(
            torch.load(encoder_path, map_location=DEVICE)
        )
        self.decoder.load_state_dict(
            torch.load(decoder_path, map_location=DEVICE)
        )

        self.encoder.eval()
        self.decoder.eval()

    def _decode_with_beam_search(self, hidden):
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
                output, next_hidden = self.decoder(dec_input, beam_hidden)
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
        """Generate a response for the given user sentence."""
        cleaned = preprocess(sentence)
        if not cleaned:
            return "Could you rephrase that? I didn't quite catch it."

        indexes = self.vocab.sentence_to_indexes(cleaned)
        indexes.append(Vocabulary.EOS_token)
        input_tensor = torch.tensor(indexes, dtype=torch.long, device=DEVICE).unsqueeze(0)

        with torch.no_grad():
            hidden = self.encoder(input_tensor)

            token_sequence = self._decode_with_beam_search(hidden)

        response_words = []
        for token_id in token_sequence[1:]:
            if token_id == Vocabulary.EOS_token:
                break
            word = self.vocab.index2word.get(token_id, "")
            if word and word not in ("<PAD>", "<SOS>", "<EOS>", "<UNK>"):
                response_words.append(word)

        if len(response_words) < MIN_RESPONSE_LEN:
            return "I can help with admissions, attendance, fees, exams, and timetable queries."

        if not response_words:
            return "I'm not sure how to respond to that. Could you try asking differently?"

        return " ".join(response_words)


# ── Quick CLI test ─────────────────────────────────────────────────────────────
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
