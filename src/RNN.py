import torch
from torch import nn
import numpy as np


class Encoder(nn.Module):
    def __init__(self, input_size, embd_size, hidden_size, num_layers) -> None:
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embd_size)
        self.gru = nn.RNN(
            input_size=embd_size, hidden_size=hidden_size, num_layers=num_layers
        )

    def forward(self, x):
        # x: L x B
        embedded = self.embedding(x)
        # embedded: L x B x E
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(
        self, input_size, embd_size, hidden_size, output_size, num_layers
    ) -> None:
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embd_size)
        self.gru = nn.RNN(embd_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # x: B -> 1 x B
        embedded = self.embedding(x.unsqueeze(0))
        # embedded: 1 x B x E
        output, hidden = self.gru(embedded, hidden)
        # output: 1 x B x H
        prediction = self.fc(output)
        # prediction: 1 x B x V -> B x V_out
        return prediction.squeeze(0), hidden


class NMT(nn.Module):
    def __init__(self, encoder, decoder, target_vocab_size):
        super(NMT, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_size = target_vocab_size

    def forward(self, source, target, tch_force=0.9):
        batch_size = source.size(1)
        target_len = target.size(0)
        _, hidden = self.encoder(source)
        outputs = torch.zeros(batch_size, target_len, self.target_size).to(
            source.device
        )

        x = target[0]
        for t in range(1, target_len):
            output, hidden = self.decoder(x, hidden)
            outputs[:, t, :] = output
            yhat = output.softmax(1).argmax(1)
            x = target[t] if np.random.random() < tch_force else yhat
        return outputs

    def translate(self, source):
        with torch.no_grad():
            _, hidden = self.encoder(source)
            outputs = [1]
            x = source[0, 0].unsqueeze(0)
            t = 0
            while x.item() != 2 and t < 50:
                output, hidden = self.decoder(x, hidden)
                x = torch.softmax(output, 1).argmax(1)
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
