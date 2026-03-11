import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Bidirectional encoder that returns token-level outputs + decoder-ready hidden state."""

    def __init__(self, vocab_size: int, hidden_size: int,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        if hidden_size % 2 != 0:
            raise ValueError("hidden_size must be even for bidirectional encoder")

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.direction_hidden = hidden_size // 2

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            hidden_size,
            self.direction_hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )

    def _combine_directions(self, state):
        # (num_layers * 2, batch, direction_hidden)
        state = state.view(self.num_layers, 2, state.size(1), self.direction_hidden)
        # Concatenate forward/backward states -> (num_layers, batch, hidden_size)
        return torch.cat((state[:, 0], state[:, 1]), dim=2)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.lstm(embedded)
        hidden = self._combine_directions(hidden)
        cell = self._combine_directions(cell)
        return outputs, (hidden, cell)


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.encoder_proj = nn.Linear(hidden_size, hidden_size)
        self.decoder_proj = nn.Linear(hidden_size, hidden_size)
        self.energy = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs, mask):
        # decoder_state: (batch, hidden)
        # encoder_outputs: (batch, src_len, hidden)
        dec = self.decoder_proj(decoder_state).unsqueeze(1)
        enc = self.encoder_proj(encoder_outputs)
        scores = self.energy(torch.tanh(enc + dec)).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights


class Decoder(nn.Module):
    """Attention decoder for token-by-token response generation."""

    def __init__(self, vocab_size: int, hidden_size: int,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.attention = BahdanauAttention(hidden_size)

        self.lstm = nn.LSTM(
            hidden_size * 2,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size * 3, vocab_size)

    def forward(self, x, hidden, encoder_outputs, src_mask=None):
        x = x.unsqueeze(1)
        embedded = self.dropout(self.embedding(x))

        context, _ = self.attention(hidden[0][-1], encoder_outputs, src_mask)
        lstm_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)

        output, hidden = self.lstm(lstm_input, hidden)
        output = output.squeeze(1)
        embedded = embedded.squeeze(1)

        prediction = self.fc(torch.cat((output, context, embedded), dim=1))
        return prediction, hidden


class Seq2Seq(nn.Module):
    """Encoder-decoder with additive attention."""

    def __init__(self, encoder: Encoder, decoder: Decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio: float = 0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, trg_len, vocab_size, device=self.device)

        src_mask = src.ne(0)
        encoder_outputs, hidden = self.encoder(src)
        dec_input = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden = self.decoder(dec_input, hidden, encoder_outputs, src_mask)
            outputs[:, t, :] = output

            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            dec_input = trg[:, t] if use_teacher else output.argmax(dim=1)

        return outputs
