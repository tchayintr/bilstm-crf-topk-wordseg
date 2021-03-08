import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.util


class Embedding(nn.Module):
    def __init__(self, input_size, embed_size):
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(input_size, embed_size)

    def forward(self, inputs):
        x = inputs[0]
        embeddings = self.embed(x)
        return embeddings


class MLP(nn.Sequential):
    def __init__(self,
                 input_size,
                 hidden_size,
                 n_layers,
                 output_size,
                 dropout,
                 activation=nn.ReLU):
        self.dropout = dropout
        self.layers = None

        layers = [None] * n_layers
        self.acts = [None] * n_layers
        hidden_size = hidden_size if hidden_size > 0 else output_size

        for i in range(n_layers):
            if i == 0:
                prev_size = input_size
                next_size = output_size if n_layers == 1 else hidden_size
                act = activation() if n_layers == 1 else nn.ReLU()

            elif i == n_layers - 1:
                prev_size = hidden_size
                next_size = output_size
                act = activation()

            else:
                prev_size = next_size = hidden_size
                act = nn.ReLU()

            layers[i] = nn.Linear(prev_size, next_size)
            self.acts[i] = act

        self.layers = layers
        super(MLP, self).__init__(*layers)

        for i in range(n_layers):
            print(
                '#    Affine {}-th layer: W={}, b={}, dropout={}, activation={}'
                .format(i, self.layers[i].weight.shape,
                        self.layers[i].bias.shape, self.dropout, self.acts[i]),
                file=sys.stderr)

    def forward(self, xs):
        hs_prev = xs
        hs = None

        for i in range(len(self.layers)):
            hs = self.acts[i](self.layers[i](F.dropout(hs_prev,
                                                       p=self.dropout)))
            hs_prev = hs

        return hs


class RNNTanh(nn.Module):
    def __init__(self,
                 embed_size,
                 hidden_size,
                 n_layers,
                 batch_first,
                 dropout,
                 bidirectional,
                 nonlinearity='tanh'):
        super(RNNTanh, self).__init__()
        self.dropout = dropout
        self.rnn = nn.RNN(input_size=embed_size,
                          hidden_size=hidden_size,
                          num_layers=n_layers,
                          nonlinearity=nonlinearity,
                          batch_first=batch_first,
                          dropout=dropout,
                          bidirectional=bidirectional)

    def forward(self, xs):
        hs, hy = self.rnn(xs)
        return F.dropout(hs, p=self.dropout), hy


class GRU(nn.Module):
    def __init__(
        self,
        embed_size,
        hidden_size,
        n_layers,
        batch_first,
        dropout,
        bidirectional,
    ):
        super(GRU, self).__init__()
        self.dropout = dropout
        self.gru = nn.GRU(input_size=embed_size,
                          hidden_size=hidden_size,
                          num_layers=n_layers,
                          batch_first=batch_first,
                          dropout=dropout,
                          bidirectional=bidirectional)

    def forward(self, xs):
        hs, hy = self.gru(xs)
        return F.dropout(hs, p=self.dropout), hy


class LSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, n_layers, batch_first, dropout,
                 bidirectional):
        super(LSTM, self).__init__()
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=n_layers,
                            batch_first=batch_first,
                            dropout=dropout,
                            bidirectional=bidirectional)

    def forward(self, xs, lengths):
        self.lstm.flatten_parameters()
        device = xs.device
        batch_first = self.lstm.batch_first

        # batch tensor sorting for sort packing
        lengths, perm_index = torch.sort(lengths, dim=0, descending=True)
        xs = xs[perm_index]
        # pack input and lengths
        xs = nn.utils.rnn.pack_padded_sequence(xs,
                                               lengths.cpu(),
                                               batch_first=batch_first)
        hs, (hy, cy) = self.lstm(xs)
        # unpack input and lengths for unsorting
        hs, lengths = nn.utils.rnn.pad_packed_sequence(hs,
                                                       batch_first=batch_first)
        perm_index_rev = torch.tensor(models.util.inverse_indices(perm_index),
                                      device=device)
        # unsort
        # hs = hs[perm_index_rev, :, :]
        hs = hs[perm_index_rev, :]

        return F.dropout(hs, p=self.dropout), (hy, cy)


class SimpleLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, n_layers, batch_first, dropout,
                 bidirectional):
        super(SimpleLSTM, self).__init__()
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=n_layers,
                            batch_first=batch_first,
                            dropout=dropout,
                            bidirectional=bidirectional)

    def forward(self, xs, lengths=None):
        self.lstm.flatten_parameters()
        hs, (hy, cy) = self.lstm(xs)
        return F.dropout(hs, p=self.dropout), (hy, cy)
