import torch
from torch import nn
import numpy as np


class NeuralMachineTranslation(nn.Module):
    def __init__(self, encoder, decoder, target_vocab_size, tch_force=0.5):
        super(NeuralMachineTranslation, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tch_force = tch_force
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
            x = target[t] if np.random.random() < self.tch_force else yhat
        return outputs

