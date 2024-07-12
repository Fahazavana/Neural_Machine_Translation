import torch
from torch import nn
from torch import optim
from torchtext.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator
import numpy as np
import spacy
import random

from torch.utils.tensorboard import SummaryWriter


spacy_en = spacy.load('en')
spacy_af = spacy.load('af')


def tokenizer_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def tokenizer_af(text):
    return [tok.text for tok in spacy_af.tokenizer(text)]


afrikaans = Field(tokenize=tokenizer_af, lower=True,
                  init_token='<s>', eos_token='</s>')
english = Field(tokenize=tokenizer_en, lower=True,
                init_token='<s>', eos_token='</s>')

train_data, val_dat, test_data = Multi30k.splits(
    exts=('.af', '.en'), fields=(afrikaans, english))


afrikaans.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)


class Encoder(nn.Module):
    def __init__(self, vocab_size, embd_size, hidden_size, num_layers, drpt) -> None:
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(drpt)
        self.embedding = nn.Embedding(vocab_size, embd_size)
        self.rnn = nn.LSTM(embd_size, hidden_size, num_layers, dropout=drpt)

    def forward(self, x):
        # x: SxB
        e = self.dropout(self.embedding(x))  # S x B x E
        _, (hidden, cell) = self.rnn(e)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, vocab_size, embd_size, hidden_size, num_layers, drpt) -> None:
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(drpt)
        self.embedding = nn.Embedding(vocab_size, embd_size)
        self.rnn = nn.LSTM(embd_size, hidden_size, num_layers, dropout=drpt)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden, cell):
        # x: B-> 1xB
        x = x.unsqueeze(0)
        e = self.dropout(self.embedding(x))  # 1 x B x E
        out, (hidden, cell) = self.rnn(e, (hidden, cell))  # 1 x B x H
        pred = self.fc(out)  # (1, B, V)
        return pred.squeeze(0), hidden, cell


class NMT(nn.Module):
    def __init__(self, encoder, decoder):
        super(NMT, self).__init__
        self.encoder = encoder
        self.decodr = decoder

    def forward(self, source, target, tch_force=0.9):
        batch_size = source.size(1)
        target_len = target.size(0)
        target_vocab = len(english.vocab)

        out = torch.zeros(target_len, batch_size, target_vocab).to(device)

        hidden, cell = self.encoder(source)

        x = target[0]
        for t in range(1, target_len):
            out_, hidden, cell = self.decoder(x, hidden, cell)
            out[t] = out_

            yhat = out_.argmax(1)
            x = target[t] if random.random() < tch_force else yhat

        return out


# Hyper-params
EPOCHS = 20
LR = 1e-3
BATCH_SIZE = 64
LOAD_MODEL = False

IN_ENCODER = len(afrikaans.vocab)
IN_DECODER = len(english.vocab)
OUTPUTSIZE = len(english.vocab)

ENCODER_E = 64
DECODER_E = 64

HIDDEN_SIZE = 512
NUM_LAYERS = 2

ENCODER_DRP = 0.5
DECODER_DRP = 0.5

writer = SummaryWriter(f"runs/loss_plot")
step = 0

train_iter, val_iter, test_iter = BucketIterator.splts((train_data, val_dat, test_data),
                                                       batch_size=BATCH_SIZE,
                                                       sort_within_batch=True,
                                                       sort_key=lambda x: len(
                                                           x.src),
                                                       device=device)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


encoder_net = Encoder(IN_ENCODER, ENCODER_E, HIDDEN_SIZE,
                      NUM_LAYERS, ENCODER_DRP).to(device)
decoder_net = Decoder(IN_DECODER, DECODER_E, HIDDEN_SIZE,
                      NUM_LAYERS, DECODER_DRP).to(device)

nmt = NMT(encoder_net, decoder_net)

pad_idx = english.vocab.stoi['<pad>']
criterion = nn.CrossEmtorpyLoss(ingore_idex=pad_idx)

optimizer = optim.Adam(nmp.parameters(), 1e-3)

for epoch in range(EPOCHS):
    print(f"Epoch [{epoch}/{EPOCHS}]")
    for batch in enumerate(train_iter):
        input_ = batch.src.to(device)
        target_ = batch.trg.to(device)

        output = nmt(input_, target_)
        output = output[1:].reshape(-1, output.shape[2])
        target_ = target_[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target_)
        loss.backward()

        torch.nn.utils.clip_grad_norm(nmp.parameters(), max_norm=1)
        optimizer.step()

        writer.add_scalar("Training_loss", loss, global_step=step)
        step +=1
