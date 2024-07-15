import torch
from torch import nn
import numpy as np


class NeuralMachineTranslation(nn.Module):
    def __init__(self, encoder, decoder, target_vocab_size):
        super(NeuralMachineTranslation, self).__init__()
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
            yhat = output.softmax(1).argmax(1)
            x = target[t] if np.random.random() < tch_force else yhat
        return outputs

    def translate(self, source, max_seq_len=30):
        with torch.no_grad():
            _, hidden = self.encoder(source)
            x = source[0, 0].unsqueeze(0)
            outputs = [1]
            t = 0
            while x.item() != 2 and t < max_seq_len:
                output, hidden = self.decoder(x, hidden)
                x = torch.argmax(output, 1)
                outputs.append(x.item())
                t += 1
        return outputs
