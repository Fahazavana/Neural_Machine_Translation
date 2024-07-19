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
        self.fc = nn.Linear(2*hidden_size, input_size)

    def forward(self, x, hidden, encoder_outputs):
        # x: B -> 1 x B
        e = self.embedding(x.unsqueeze(0))
        # e: 1 x B x E
        decoded, hidden = self.rnn(e, hidden)
        # out: 1 x B x H
        ##############################################################################################
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        decoded = decoded.permute(1, 0, 2)
        attn_scores = torch.einsum("blh,bih->bl", encoder_outputs, decoded) / np.sqrt(
            self.rnn.hidden_size
        )
        alpha = attn_scores.softmax(
            dim=1
        )  # Alpha: B x L (L - encoder output sequence length)
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs)  # Context: 1 x B x H
        output = torch.cat(
            (decoded.permute(1, 0, 2), context.permute(1, 0, 2)), dim=-1
        )  # Concatenate on hidden size dimension
        ##############################################################################################
        prediction = self.fc(output)
        # pred: 1 x B x V -> B x V_out
        return prediction.squeeze(0), hidden


class LSTMANMT(nn.Module):
    def __init__(self, encoder, decoder, target_vocab_size, tch_force=0.9):
        super(LSTMANMT, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_size = target_vocab_size
        self.tch_force = tch_force

    def forward(self, source, target):
        target_len, batch_size = target.shape
        encoder_output , hidden = self.encoder(source)
        outputs = torch.zeros(batch_size, target_len, self.target_size).to(
            source.device
        )
        x = target[0]
        hidden = torch.zeros_like(hidden[0]), torch.zeros_like(hidden[1])
        for t in range(1, target_len):
            output, hidden = self.decoder(x, hidden, encoder_output)
            outputs[:, t, :] = output
            yhat = output.argmax(1).detach()
            x = target[t] if np.random.random() < self.tch_force else yhat
        return outputs
