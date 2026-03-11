import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Reads the input sentence and produces a context (hidden state)."""

    def __init__(self, vocab_size: int, hidden_size: int,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(
            hidden_size, hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.dropout(self.embedding(x))   # (batch, seq_len, hidden)
        _, hidden = self.lstm(embedded)               # hidden: tuple of (num_layers, batch, hidden)
        return hidden


class Decoder(nn.Module):
    """Generates the response one word at a time."""

    def __init__(self, vocab_size: int, hidden_size: int,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(
            hidden_size, hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden):
        # x: (batch,)  — single token index per sample
        x = x.unsqueeze(1)                            # (batch, 1)
        embedded = self.dropout(self.embedding(x))    # (batch, 1, hidden)
        output, hidden = self.lstm(embedded, hidden)  # (batch, 1, hidden)
        prediction = self.fc(output.squeeze(1))       # (batch, vocab_size)
        return prediction, hidden


class Seq2Seq(nn.Module):
    """Combines Encoder and Decoder into a full sequence-to-sequence model."""

    def __init__(self, encoder: Encoder, decoder: Decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio: float = 0.5):
        """
        src : (batch, src_len)  — input sentence tokens
        trg : (batch, trg_len)  — expected response tokens (starts with <SOS>)
        """
        batch_size = src.size(0)
        trg_len = trg.size(1)
        vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, trg_len, vocab_size, device=self.device)

        hidden = self.encoder(src)

        # First decoder input is the <SOS> token
        dec_input = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden = self.decoder(dec_input, hidden)
            outputs[:, t, :] = output

            # Use teacher forcing or model's own prediction
            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            dec_input = trg[:, t] if use_teacher else output.argmax(dim=1)

        return outputs
