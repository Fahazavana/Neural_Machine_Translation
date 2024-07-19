from torch import nn
from .NeuralMachineTranslation import NeuralMachineTranslation
from torch import nn

from .NeuralMachineTranslation import NeuralMachineTranslation


class Encoder(nn.Module):
    def __init__(self, input_size, embd_size, hidden_size, num_layers, type='RNN') -> None:
        super(Encoder, self).__init__()
        if type == 'RNN':
            self.rnn = nn.RNN(embd_size, hidden_size, num_layers)
        elif type == 'GRU':
            self.rnn = nn.GRU(embd_size, hidden_size, num_layers)
        else:
            raise ValueError(f"type must be 'RNN' or 'GRU', got {type}")
        self.embedding = nn.Embedding(input_size, embd_size)

    def forward(self, x):
        # x: L x B
        embedded = self.embedding(x)
        # embedded: L x B x E
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, input_size, embd_size, hidden_size, num_layers, type='RNN') -> None:
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if type == 'RNN':
            self.rnn = nn.RNN(embd_size, hidden_size, num_layers)
        elif type == 'GRU':
            self.rnn = nn.GRU(embd_size, hidden_size, num_layers)
        else:
            raise ValueError(f"type must be 'RNN' or 'GRU', got {type}")
        self.embedding = nn.Embedding(input_size, embd_size)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, hidden):
        # x: B -> 1 x B
        embedded = self.embedding(x.unsqueeze(0))
        # embedded: 1 x B x E
        output, hidden = self.rnn(embedded, hidden)
        # output: 1 x B x H
        prediction = self.fc(output)
        # prediction: 1 x B x V -> B x V_out
        return prediction.squeeze(0), hidden


class RNN_NMT(NeuralMachineTranslation):
    def __init__(self, encoder, decoder, target_vocab_size, tch_force=0.9):
        super(RNN_NMT, self).__init__(encoder, decoder, target_vocab_size, tch_force)


class GRU_NMT(NeuralMachineTranslation):
    def __init__(self, encoder, decoder, target_vocab_size, tch_force=0.9):
        super(GRU_NMT, self).__init__(encoder, decoder, target_vocab_size, tch_force)
