import torch
from torch import nn
import numpy as np


class Encoder(nn.Module):
    def __init__(self, input_size, embd_size, hidden_size, num_layers) -> None:
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embd_size)
        self.rnn = nn.LSTM(
            input_size=embd_size, hidden_size=hidden_size, num_layers=num_layers
        )

    def forward(self, x):
        # x: L x B
        e = self.embedding(x)
        # e: L x B x E
        _, (hidden, cell) = self.rnn(e)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(
        self, input_size, embd_size, hidden_size, output_size, num_layers
    ) -> None:
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embd_size)
        self.rnn = nn.LSTM(embd_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        # x: B -> 1 x B
        e = self.embedding(x.unsqueeze(0))
        # e: 1 x B x E
        out, (hidden, cell) = self.rnn(e, (hidden, cell))
        # out: 1 x B x H
        pred = self.fc(out)
        # pred: 1 x B x V -> B x V_out
        return pred.squeeze(0), hidden, cell


class NMT(nn.Module):
    def __init__(self, encoder, decoder, target_vocab_size):
        super(NMT, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_size = target_vocab_size

    def forward(self, source, target, tch_force=0.95):
        batch_size = source.size(1)
        target_len = target.size(0)
        hidden, cell = self.encoder(source)
        outputs = torch.zeros(batch_size, target_len, self.target_size).to(
            source.device
        )

        x = target[0]
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[:, t, :] = output
            yhat = output.argmax(1).detach()
            x = target[t] if np.random.random() < tch_force else yhat
        return outputs

    def translate(self, source):
        with torch.no_grad():
            hidden, cell = self.encoder(source)
            x = source[0, 0].unsqueeze(0)
            t = 0
            outputs = [x.item()]
            while x.item() != 2 and t < 50:
                output, hidden, cell = self.decoder(x, hidden, cell)
                x = torch.argmax(output, 1)
                outputs.append(x.item())
                t += 1
        return outputs


def translate(model, text, source, target, device):
    text = [source.stoi[word] for word in text.strip().split()]
    text = torch.tensor(text, dtype=torch.long).unsqueeze(1)
    text = text.to(device)
    out = model.translate(text)
    out = [target.itos[idx] for idx in out]
    return " ".join(out)
