import sys

from classifiers.classifier import Classifier
import common
import constants
import models.util


class SequenceTagger(Classifier):
    def __init__(self, predictor, task=constants.TASK_SEG):
        super(SequenceTagger, self).__init__(predictor=predictor)
        self.task = task

    def forward(self, *inputs, train=False):
        ret = self.predictor(*inputs)
        return ret

    def decode(self, *inputs):
        ys = self.predictor.decode(*inputs)
        return ys

    def change_dropout_ratio(self, dropout_ratio):
        self.change_embed_dropout_ratio(dropout_ratio)
        self.change_rnn_dropout_ratio(dropout_ratio)
        self.change_mlp_dropout_ratio(dropout_ratio)

    def change_embed_dropout_ratio(self, dropout_ratio):
        self.predictor.embed_dropout = dropout_ratio
        print('Set {} dropout ratio to {}'.format(
            'embedding', self.predictor.embed_dropout),
              file=sys.stderr)

    def change_rnn_dropout_ratio(self, dropout_ratio):
        self.predictor.rnn.dropout = dropout_ratio
        print('Set {} dropout ratio to {}'.format('RNN',
                                                  self.predictor.rnn.dropout),
              file=sys.stderr)

    def change_mlp_dropout_ratio(self, dropout_ratio):
        self.predictor.mlp.dropout = dropout_ratio
        print('Set {} dropout ratio to {}'.format('MLP', dropout_ratio),
              file=sys.stderr)

    def change_crf_top_k_argument(self, top_k):
        self.predictor.crf_top_k = top_k
        print('Set {} top k to {}'.format('CRF', top_k), file=sys.stderr)

    def load_pretrained_embedding_layer(self,
                                        dic,
                                        external_unigram_model,
                                        finetuning=False):
        usage = self.predictor.pretrained_embed_usage

        if external_unigram_model:
            id2unigram = dic.tables[constants.UNIGRAM].id2str
            if usage == models.util.ModelUsage.INIT:
                models.util.load_pretrained_embedding_layer(
                    id2unigram,
                    self.predictor.unigram_embed,
                    external_unigram_model,
                    finetuning=finetuning)
            elif usage == models.util.ModelUsage.ADD or models.util.ModelUsage.CONCAT:
                models.util.load_pretrained_embedding_layer(
                    id2unigram,
                    self.predictor.pretrained_unigram_embed,
                    external_unigram_model,
                    finetuning=finetuning)

    def grow_embedding_layers(self,
                              dic_grown,
                              external_unigram_model=None,
                              train=True):
        if (self.predictor.pretrained_embed_usage == models.util.ModelUsage.ADD
                or self.predictor.pretrained_embed_usage
                == models.util.ModelUsage.CONCAT):
            pretrained_unigram_embed = self.predictor.pretrained_unigram_embed
        else:
            pretrained_unigram_embed = None

        id2unigram_grown = dic_grown.tables[constants.UNIGRAM].id2str
        n_unigrams_org = self.predictor.unigram_embed.weight.shape[0]
        n_unigrams_grown = len(id2unigram_grown)
        models.util.grow_embedding_layers(
            n_unigrams_org,
            n_unigrams_grown,
            self.predictor.unigram_embed,
            pretrained_unigram_embed,
            external_unigram_model,
            id2unigram_grown,
            self.predictor.pretrained_embed_usage,
            train=train)

    def grow_inference_layers(self, dic_grown):
        n_labels_org = self.predictor.mlp.layers[-1].weight.shape[0]
        if common.is_segmentation_task(self.task):
            n_labels_grown = len(dic_grown.tables[constants.SEG_LABEL].id2str)
        models.util.grow_MLP(n_labels_org, n_labels_grown,
                             self.predictor.mlp.layers[-1])
        if self.predictor.use_crf:
            models.util.grow_crf_layer(n_labels_org, n_labels_grown,
                                       self.predictor.crf)
