"""
chatbot.py  —  Inference engine for the trained Seq2Seq chatbot.
"""

import os
import torch

from vocabulary import Vocabulary, preprocess
from model import Encoder, Decoder

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_DIR        = "saved_model"
HIDDEN_SIZE      = 256
NUM_LAYERS       = 2
MAX_RESPONSE_LEN = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Chatbot:
    def __init__(self):
        vocab_path   = os.path.join(MODEL_DIR, "vocab.pkl")
        encoder_path = os.path.join(MODEL_DIR, "encoder.pt")
        decoder_path = os.path.join(MODEL_DIR, "decoder.pt")

        if not all(os.path.exists(p) for p in [vocab_path, encoder_path, decoder_path]):
            raise FileNotFoundError(
                "Trained model not found. Run 'python train.py' first."
            )

        self.vocab   = Vocabulary.load(vocab_path)
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

            dec_input    = torch.tensor([Vocabulary.SOS_token], device=DEVICE)
            response_words = []

            for _ in range(MAX_RESPONSE_LEN):
                output, hidden = self.decoder(dec_input, hidden)
                top_token = output.argmax(dim=1)
                token_id  = top_token.item()

                if token_id == Vocabulary.EOS_token:
                    break

                word = self.vocab.index2word.get(token_id, "")
                if word and word not in ("<PAD>", "<SOS>", "<EOS>", "<UNK>"):
                    response_words.append(word)

                dec_input = top_token

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
