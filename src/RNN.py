import torch
from torch import nn
import numpy as np
from .NeuralMachineTranslation import NeuralMachineTranslation

class Encoder(nn.Module):
    def __init__(self, input_size, embd_size, hidden_size, num_layers) -> None:
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, embd_size)
        self.gru = nn.RNN(embd_size, hidden_size, num_layers)

    def forward(self, x):
        # x: L x B
        embedded = self.embedding(x)
        # embedded: L x B x E
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(
        self, input_size, embd_size, hidden_size, num_layers
    ) -> None:
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embd_size)
        self.gru = nn.RNN(embd_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, hidden):
        # x: B -> 1 x B
        embedded = self.embedding(x.unsqueeze(0))
        # embedded: 1 x B x E
        output, hidden = self.gru(embedded, hidden)
        # output: 1 x B x H
        prediction = self.fc(output)
        # prediction: 1 x B x V -> B x V_out
        return prediction.squeeze(0), hidden


class RNN_NMT(NeuralMachineTranslation):
	def __init__(self, encoder, decoder, target_vocab_size):
		super(RNN_NMT, self).__init__(encoder, decoder, target_vocab_size)
