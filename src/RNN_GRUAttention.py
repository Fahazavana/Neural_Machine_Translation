import numpy as np
import torch
import torch.nn as nn

from .NeuralMachineTranslation import NeuralMachineTranslation


class Encoder(nn.Module):
    def __init__(
        self,
        input_size,
        embd_size,
        hidden_size,
        num_layers,
        type="RNN"
    ) -> None:
        super(Encoder, self).__init__()
        if type == "RNN":
            self.rnn = nn.RNN(
                embd_size, hidden_size, num_layers)
        elif type == "GRU":
            self.rnn = nn.GRU(
                embd_size, hidden_size, num_layers)
        else:
            raise ValueError(f"type must be 'RNN' or 'GRU', got {type}")
        self.embedding = nn.Embedding(input_size, embd_size)

    def forward(self, x):
        # x: L x B
        embedded = self.embedding(x)
        # embedded: L x B x E
        output, hidden = self.rnn(embedded)
        return output, hidden


class Decoder(nn.Module):
    def __init__(
        self,
        input_size,
        embd_size,
        hidden_size,
        num_layers,
        type="RNN",
    ) -> None:
        super(Decoder, self).__init__()
        if type == "RNN":
            self.rnn = nn.RNN(
                embd_size, hidden_size, num_layers
            )
        elif type == "GRU":
            self.rnn = nn.GRU(
                embd_size, hidden_size, num_layers)
        else:
            raise ValueError(f"type must be 'RNN' or 'GRU', got {type}")
        self.embedding = nn.Embedding(input_size, embd_size)
        self.fc = nn.Linear(
            hidden_size * 2, input_size
        )  

    def forward(self, x, hidden, encoder_outputs):
        # x: B -> 1 x B
        embedded = self.embedding(x.unsqueeze(0))  # Embedded: 1 x B x E
        decoded, hidden = self.rnn(embedded, hidden)  # Output: 1 x B x H
        ##############################################################################################
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        decoded = decoded.permute(1, 0, 2)
        attn_scores = torch.einsum("blh,bih->bl", encoder_outputs, decoded) / np.sqrt(
            self.rnn.hidden_size
        )
        alpha = attn_scores.softmax(
            dim=1
        )  # Alpha: B x L 
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs)  # Context: 1 x B x H
        output = torch.cat(
            (decoded.permute(1, 0, 2), context.permute(1, 0, 2)), dim=-1
        )  
        ##############################################################################################
        prediction = self.fc(output)  # Prediction: 1 x B x V -> B x V_out
        return prediction.squeeze(0), hidden


class Attention_NMT(NeuralMachineTranslation):
    def __init__(self, encoder, decoder, target_vocab_size, tch_force=0.9):
        super(Attention_NMT, self).__init__(encoder, decoder, target_vocab_size, tch_force)

    def forward(self, source, target):
        target_len, batch_size = target.shape
        encoder_output, hidden = self.encoder(source)
        outputs = torch.zeros(batch_size, target_len, self.target_size).to(
            source.device
        )
        x = target[0]
        
        for t in range(1, target_len):
            output, hidden = self.decoder(x, hidden, encoder_output)
            outputs[:, t, :] = output
            yhat = output.softmax(1).argmax(1)
            x = target[t] if np.random.random() < self.tch_force else yhat
        return outputs
    
class RNNAtt(Attention_NMT):
	def __init__(self, encoder, decoder, target_vocab_size, tch_force=0.9):
		super(RNNAtt, self).__init__(encoder, decoder, target_vocab_size, tch_force)

class GRUAtt(NeuralMachineTranslation):
    def __init__(self, encoder, decoder, target_vocab_size, tch_force=0.9):
        super(GRUAtt, self).__init__(encoder, decoder, target_vocab_size, tch_force)

