import torch
from torch import nn
import numpy as np


class Encoder(nn.Module):
    def __init__(self, input_size, embd_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, embd_size)
        self.rnn = nn.LSTM(embd_size, hidden_size, num_layers)

    def forward(self, x):
        # x: L x B
        e = self.embedding(x)
        # e: L x B x E
        output, hidden = self.rnn(e)
        return output, hidden


class Decoder(nn.Module):
    def __init__(self, input_size, embd_size, hidden_size, num_layers):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(input_size, embd_size)
        self.rnn = nn.LSTM(embd_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, hidden):
        # x: B -> 1 x B
        e = self.embedding(x.unsqueeze(0))
        # e: 1 x B x E
        out, hidden = self.rnn(e, hidden)
        # out: 1 x B x H
        prediction = self.fc(out)
        # pred: 1 x B x V -> B x V_out
        return prediction.squeeze(0), hidden


class LSTM_NMT(nn.Module):
    def __init__(self, encoder, decoder, target_vocab_size):
        super(LSTM_NMT, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_size = target_vocab_size

    def forward(self, source, target, tch_force=0.9):
        target_len, batch_size = target.shape
        _, hidden = self.encoder(source)
        outputs = torch.zeros(batch_size, target_len, self.target_size).to(
            source.device
        )
        x = target[0]
        for t in range(1, target_len):
            output, hidden = self.decoder(x, hidden)
            outputs[:, t, :] = output
            yhat = output.argmax(1).detach()
            x = target[t] if np.random.random() < tch_force else yhat
        return outputs
