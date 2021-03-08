from allennlp.modules.conditional_random_field import ConditionalRandomField
from allennlp.nn.util import get_mask_from_sequence_lengths
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import sys

import models.util
from models.util import ModelUsage
from models.common import MLP


class RNNTagger(nn.Module):
    def __init__(self,
                 n_vocab,
                 unigram_embed_size,
                 rnn_unit_type,
                 rnn_bidirection,
                 rnn_batch_first,
                 rnn_n_layers,
                 rnn_hidden_size,
                 mlp_n_layers,
                 mlp_hidden_size,
                 n_labels,
                 use_crf=True,
                 crf_top_k=1,
                 embed_dropout=0.0,
                 rnn_dropout=0.0,
                 mlp_dropout=0.0,
                 pretrained_unigram_embed_size=0,
                 pretrained_embed_usage=ModelUsage.NONE):
        super(RNNTagger, self).__init__()
        self.n_vocab = n_vocab
        self.unigram_embed_size = unigram_embed_size

        self.rnn_unit_type = rnn_unit_type
        self.rnn_bidirection = rnn_bidirection
        self.rnn_batch_first = rnn_batch_first
        self.rnn_n_layers = rnn_n_layers
        self.rnn_hidden_size = rnn_hidden_size

        self.mlp_n_layers = mlp_n_layers
        self.mlp_hidden_size = mlp_hidden_size
        self.n_labels = n_labels
        self.use_crf = use_crf
        self.crf_top_k = crf_top_k

        self.embed_dropout = embed_dropout
        self.rnn_dropout = rnn_dropout
        self.mlp_dropout = mlp_dropout

        self.pretrained_unigram_embed_size = pretrained_unigram_embed_size
        self.pretrained_embed_usage = pretrained_embed_usage

        self.unigram_embed = None
        self.pretrained_unigram_embed = None
        self.rnn = None
        self.mlp = None
        self.crf = None
        self.cross_entropy_loss = None

        print('### Parameters', file=sys.stderr)

        # embeddings layer(s)

        print('# Embedding dropout ratio={}'.format(self.embed_dropout),
              file=sys.stderr)
        self.unigram_embed, self.pretrained_unigram_embed = models.util.construct_embeddings(
            n_vocab, unigram_embed_size, pretrained_unigram_embed_size,
            pretrained_embed_usage)
        if self.pretrained_embed_usage != ModelUsage.NONE:
            print('# Pretrained embedding usage: {}'.format(
                self.pretrained_embed_usage),
                  file=sys.stderr)
        print('# Unigram embedding matrix: W={}'.format(
            self.unigram_embed.weight.shape),
              file=sys.stderr)
        embed_size = self.unigram_embed.weight.shape[1]
        if self.pretrained_unigram_embed is not None:
            if self.pretrained_embed_usage == ModelUsage.CONCAT:
                embed_size += self.pretrained_unigram_embed_size
                print('# Pretrained unigram embedding matrix: W={}'.format(
                    self.pretrained_unigram_embed.weight.shape),
                      file=sys.stderr)

        # recurrent layers

        self.rnn_unit_type = rnn_unit_type
        self.rnn = models.util.construct_RNN(unit_type=rnn_unit_type,
                                             embed_size=embed_size,
                                             hidden_size=rnn_hidden_size,
                                             n_layers=rnn_n_layers,
                                             batch_first=rnn_batch_first,
                                             dropout=rnn_dropout,
                                             bidirectional=rnn_bidirection)
        rnn_output_size = rnn_hidden_size * (2 if rnn_bidirection else 1)

        # MLP

        print('# MLP', file=sys.stderr)
        mlp_in = rnn_output_size
        self.mlp = MLP(input_size=mlp_in,
                       hidden_size=mlp_hidden_size,
                       n_layers=mlp_n_layers,
                       output_size=n_labels,
                       dropout=mlp_dropout,
                       activation=nn.Identity)

        # Inference layer (CRF/softmax)

        if self.use_crf:
            self.crf = ConditionalRandomField(n_labels)
            print('# CRF cost: {}'.format(self.crf.transitions.shape),
                  file=sys.stderr)
        else:
            self.softmax_cross_entropy = nn.CrossEntropyLoss()

    """
    us: batch of unigram sequences
    ls: batch of label sequences
    """

    # unigram and label
    def forward(self, us, ls=None, calculate_loss=True, decode=False):
        lengths = self.extract_lengths(us)
        us, ls = self.pad_features(us, ls)
        xs = self.extract_features(us)
        rs = self.rnn_output(xs, lengths)
        ys = self.mlp(rs)
        loss, ps = self.predict(ys,
                                ls=ls,
                                lengths=lengths,
                                calculate_loss=calculate_loss,
                                decode=decode)
        return loss, ps

    def extract_lengths(self, ts):
        device = ts[0].device
        return torch.tensor([t.shape[0] for t in ts], device=device)

    def pad_features(self, us, ls):
        batch_first = self.rnn_batch_first
        us = pad_sequence(us, batch_first=batch_first)
        ls = pad_sequence(ls, batch_first=batch_first) if ls else None

        return us, ls

    def extract_features(self, us):
        xs = []

        for u in us:
            ue = self.unigram_embed(u)
            if self.pretrained_unigram_embed is not None:
                if self.pretrained_embed_usage == ModelUsage.ADD:
                    pe = self.pretrained_unigram_embed(u)
                    ue = ue + pe
                elif self.pretrained_embed_usage == ModelUsage.CONCAT:
                    pe = self.pretrained_unigram_embed(u)
                    ue = torch.cat((ue, pe), 1)
            ue = F.dropout(ue, p=self.embed_dropout)
            xe = ue
            xs.append(xe)

        if self.rnn_batch_first:
            xs = torch.stack(xs, dim=0)
        else:
            xs = torch.stack(xs, dim=1)

        return xs

    def rnn_output(self, xs, lengths=None):
        if self.rnn_unit_type == 'lstm':
            hs, (hy, cy) = self.rnn(xs, lengths)
        else:
            hs, hy = self.rnn(xs)
        return hs

    def predict(self,
                rs,
                ls=None,
                lengths=None,
                calculate_loss=True,
                decode=False):
        if self.crf:
            return self.predict_crf(rs, ls, lengths, calculate_loss, decode)
        else:
            return self.predict_softmax(rs, ls, calculate_loss)

    def predict_softmax(self, ys, ls=None, calculate_loss=True):
        ps = []
        loss = torch.tensor(0, dtype=torch.float, device=ys.device)
        if ls is None:
            ls = [None] * len(ys)
        for y, l in zip(ys, ls):
            if calculate_loss:
                loss += self.softmax_cross_entropy(y, l)
            ps.append([torch.argmax(yi.data) for yi in y])

        return loss, ps

    def predict_crf(self,
                    hs,
                    ls=None,
                    lengths=None,
                    calculate_loss=True,
                    decode=False):
        device = hs.device
        if lengths is None:
            lengths = torch.tensor([h.shape[0] for h in hs], device=device)
        mask = get_mask_from_sequence_lengths(lengths, max_length=max(lengths))
        if not decode or self.crf_top_k == 1:
            ps = self.crf.viterbi_tags(hs, mask)
            ps, score = zip(*ps)
        else:
            ps = []
            psks = self.crf.viterbi_tags(hs, mask, top_k=self.crf_top_k)
            for psk in psks:
                psk, score = zip(*psk)
                ps.append(psk)

        if calculate_loss:
            log_likelihood = self.crf(hs, ls, mask)
            loss = -1 * log_likelihood / len(lengths)
        else:
            loss = torch.tensor(np.array(0), dtype=torch.float, device=device)

        return loss, ps

    def decode(self, us):
        with torch.no_grad():
            _, ps = self.forward(us, calculate_loss=False, decode=True)
        return ps


def construct_RNNTagger(
    n_vocab,
    unigram_embed_size,
    rnn_unit_type,
    rnn_bidirection,
    rnn_batch_first,
    rnn_n_layers,
    rnn_hidden_size,
    mlp_n_layers,
    mlp_hidden_size,
    n_labels,
    use_crf=True,
    crf_top_k=1,
    rnn_dropout=0.0,
    embed_dropout=0.0,
    mlp_dropout=0.0,
    pretrained_unigram_embed_size=0,
    pretrained_embed_usage=ModelUsage.NONE,
):

    tagger = RNNTagger(
        n_vocab=n_vocab,
        unigram_embed_size=unigram_embed_size,
        rnn_unit_type=rnn_unit_type,
        rnn_bidirection=rnn_bidirection,
        rnn_batch_first=rnn_batch_first,
        rnn_n_layers=rnn_n_layers,
        rnn_hidden_size=rnn_hidden_size,
        mlp_n_layers=mlp_n_layers,
        mlp_hidden_size=mlp_hidden_size,
        n_labels=n_labels,
        use_crf=use_crf,
        crf_top_k=crf_top_k,
        embed_dropout=embed_dropout,
        rnn_dropout=rnn_dropout,
        mlp_dropout=mlp_dropout,
        pretrained_unigram_embed_size=pretrained_unigram_embed_size,
        pretrained_embed_usage=pretrained_embed_usage)
    return tagger
