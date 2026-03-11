import pickle
import re


class Vocabulary:
    PAD_token = 0   # Padding
    SOS_token = 1   # Start of sentence
    EOS_token = 2   # End of sentence
    UNK_token = 3   # Unknown word

    def __init__(self, name="chatbot"):
        self.name = name
        self.word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.word_count = {}
        self.n_words = 4

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word_count[word] = 1
            self.n_words += 1
        else:
            self.word_count[word] = self.word_count.get(word, 0) + 1

    def sentence_to_indexes(self, sentence):
        return [
            self.word2index.get(word, self.UNK_token)
            for word in sentence.split()
        ]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


def preprocess(sentence: str) -> str:
    """Lowercase, strip punctuation (keep spaces), collapse whitespace."""
    sentence = sentence.lower().strip()
    sentence = re.sub(r"[^a-z\s]", " ", sentence)
    sentence = re.sub(r"\s+", " ", sentence).strip()
    return sentence
