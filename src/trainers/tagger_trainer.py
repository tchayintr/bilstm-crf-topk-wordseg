import sys
import torch

import classifiers.sequence_tagger
import common
import constants
from data_loaders import segmentation_data_loader
from evaluators.common import FMeasureEvaluator
import models.tagger
from trainers import trainer
from trainers.trainer import Trainer
import util


class TaggerTrainerBase(Trainer):
    def __init__(self, args, logger=sys.stderr):
        super().__init__(args, logger)

    def show_data_info(self, data_type):
        dic = self.dic_val if data_type == 'valid' else self.dic
        self.log('### {} dic'.format(data_type))
        self.log('Number of tokens: {}'.format(
            len(dic.tables[constants.UNIGRAM])))
        self.log()

    def show_training_data(self):
        train = self.train
        valid = self.valid

        self.log('### Loaded data')
        self.log('# train: {} ... {}\n'.format(train.inputs[0][0],
                                               train.inputs[0][-1]))
        self.log('# train_gold: {} ... {}\n'.format(train.outputs[0][0],
                                                    train.outputs[0][-1]))
        t2i_tmp = list(self.dic.tables[constants.UNIGRAM].str2id.items())
        self.log('# token2id: {} ... {}\n'.format(t2i_tmp[:10],
                                                  t2i_tmp[len(t2i_tmp) - 10:]))

        if self.dic.has_table(constants.SEG_LABEL):
            id2seg = {
                v: k
                for k, v in self.dic.tables[
                    constants.SEG_LABEL].str2id.items()
            }
            self.log('# label_set: {}\n'.format(id2seg))

        self.report('[INFO] vocab: {}'.format(
            len(self.dic.tables[constants.UNIGRAM])))
        self.report('[INFO] data length: train={} valid={}'.format(
            len(train.inputs[0]),
            len(valid.inputs[0]) if valid else 0))

    def gen_inputs(self, data, ids, evaluate=True):
        device = torch.device(
            constants.CUDA_DEVICE) if self.args.gpu >= 0 else torch.device(
                constants.CPU_DEVICE)

        us = [
            torch.tensor(data.inputs[0][j], dtype=int, device=device)
            for j in ids
        ]
        ls = [
            torch.tensor(data.outputs[0][j], dtype=int, device=device)
            for j in ids
        ] if evaluate else None

        if evaluate:
            return us, ls
        else:
            return us

    def decode(self, rdata, file=sys.stdout):
        n_ins = len(rdata.inputs[0])
        org_tokens = rdata.orgdata[0]

        timer = util.Timer()
        timer.start()
        for ids in trainer.batch_generator(n_ins,
                                           batch_size=self.args.batch_size,
                                           shuffle=False):
            inputs = self.gen_inputs(rdata, ids, evaluate=False)
            ot = [org_tokens[j] for j in ids]
            self.decode_batch(*[inputs], org_tokens=ot, file=file)
        timer.stop()

        print(
            'Parsed {} sentences. Elapsed time: {:.4f} sec (total) / {:.4f} sec (per sentence)'
            .format(n_ins, timer.elapsed, timer.elapsed / n_ins),
            file=sys.stderr)

    def run_interactive_mode(self):
        print('Please input text or type \'q\' to quit this mode:')
        while True:
            line = sys.stdin.readline().rstrip(' \t\n')
            if len(line) == 0:
                continue
            elif line == 'q':
                break

            rdata = self.data_loader.parse_commandline_input(line, self.dic)
            inputs = self.gen_inputs(rdata, [0], evaluate=False)
            ot = rdata.orgdata[0]

            self.decode_batch(*[inputs], org_tokens=ot)

    def run_eval_mode(self):
        super().run_eval_mode()

    def run_eval_mode_for_stat_test(self):
        classifier = self.classifier.copy()
        self.update_model(classifier=classifier, dic=self.dic)
        classifier.change_dropout_ratio(0)

        data = self.test
        n_ins = len(data.inputs[0])
        n_sen = 0
        total_counts = None

        for ids in trainer.batch_generator(n_ins,
                                           batch_size=self.args.batch_size,
                                           shuffle=False):
            inputs = self.gen_inputs(data, ids)
            xs = inputs[0]
            n_sen += len(xs)
            gls = inputs[self.label_begin_index]

            with torch.no_grad():
                ret = classifier.predictor(*inputs)
            pls = ret[1]

            for gl, pl in zip(gls, pls):
                for gli, pli in zip(gl, pl):
                    print('{}'.format(1 if int(gli) == int(pli) else 0))
                print()

        print('Finished', n_sen, file=sys.stderr)

    def convert_to_valid_BIES_seq(self, y_str):
        y_str2 = y_str.copy()

        for j in range(len(y_str)):
            prv = y_str2[j - 1] if j >= 1 else None
            crt = y_str[j]
            nxt = y_str[j + 1] if j <= len(y_str) - 2 else None

            # invalid I or E assigned for a first token
            if ((crt[0] == 'I' or crt[0] == 'E')
                    and (prv is None or prv[0] == 'S' or prv[0] == 'E')):
                if nxt == 'I' + crt[1:] or nxt == 'E' + crt[1:]:
                    y_str2[j] = 'B' + crt[1:]
                else:
                    y_str2[j] = 'S' + crt[1:]

            # invalid B or I assignied for a last token
            elif ((crt[0] == 'B' or crt[0] == 'I')
                  and (nxt is None or nxt[0] == 'B' or nxt[0] == 'S')):
                if (prv == 'B' + crt[1:] or prv == 'I' + crt[1:]):
                    y_str2[j] = 'E' + crt[1:]
                else:
                    y_str2[j] = 'S' + crt[1:]

        return y_str2

    def decode_batch(self, *inputs, org_tokens=None, file=sys.stdout):
        ys = self.classifier.decode(*inputs)
        id2label = self.dic.tables[constants.SEG_LABEL].id2str

        if self.args.crf_top_k < 2:
            for x_str, y in zip(org_tokens, ys):
                y_str = [id2label[int(yi)] for yi in y]
                y_str = self.convert_to_valid_BIES_seq(y_str)
                res = [
                    '{}{}'.format(
                        xi_str, self.args.output_token_delim if
                        (yi_str.startswith('E')
                         or yi_str.startswith('S')) else '')
                    for xi_str, yi_str in zip(x_str, y_str)
                ]
                res = ''.join(res).rstrip(' ')
                print(res, file=file)
        else:
            ysks = ys
            for x_str, ysk in zip(org_tokens, ysks):
                for y in ysk:
                    y_str = [id2label[int(yi)] for yi in y]
                    y_str = self.convert_to_valid_BIES_seq(y_str)
                    res = [
                        '{}{}'.format(
                            xi_str, self.args.output_token_delim if
                            (yi_str.startswith('E')
                             or yi_str.startswith('S')) else '')
                        for xi_str, yi_str in zip(x_str, y_str)
                    ]
                    res = ''.join(res).rstrip(' ')
                    print(res, end=constants.SL_COLUMN_DELIM, file=file)
                print('', file=file)

    def load_external_dictionary(self):
        if self.args.external_dic_path:
            edic_path = self.args.external_dic_path
            self.dic = segmentation_data_loader.load_external_dictionary(
                edic_path, dic=self.dic)
            self.log('Load external dictionary: {}'.format(edic_path))
            self.log('Num of unigrams: {}'.format(
                len(self.dic.tables[constants.UNIGRAM])))
            self.log('')


class TaggerTrainer(TaggerTrainerBase):
    def __init__(self, args, logger=sys.stderr):
        super().__init__(args, logger)
        self.unigram_embed_model = None
        self.label_begin_index = 1

    def load_external_embedding_models(self):
        if self.args.unigram_embed_model_path:
            self.unigram_embed_model = trainer.load_embedding_model(
                self.args.unigram_embed_model_path)

    def init_model(self):
        super().init_model()
        if self.unigram_embed_model:
            self.classifier.load_pretrained_embedding_layer(
                self.dic, self.unigram_embed_model, finetuning=True)

    def load_model(self):
        super().load_model()
        if self.args.execute_mode == 'train':
            if 'embed_dropout' in self.hparams:
                self.classifier.change_embed_dropout_ratio(
                    self.hparams['embed_dropout'])
            if 'rnn_dropout' in self.hparams:
                self.classifier.change_rnn_dropout_ratio(
                    self.hparams['rnn_dropout'])
            if 'mlp_dropout' in self.hparams:
                self.classifier.change_mlp_dropout_ratio(
                    self.hparams['mlp_dropout'])
        else:
            self.classifier.change_dropout_ratio(0)

        if ((self.args.execute_mode == 'decode'
             or self.args.execute_mode == 'interactive')
                and self.hparams['inference_layer'] == 'crf'
                and self.args.crf_top_k > 1):
            self.classifier.change_crf_top_k_argument(self.args.crf_top_k)
        print('', file=sys.stderr)

    def update_model(self, classifier=None, dic=None, train=False):
        if not classifier:
            classifier = self.classifier
        if not dic:
            dic = self.dic

        if (self.args.execute_mode == 'train'
                or self.args.execute_mode == 'eval'
                or self.args.execute_mode == 'decode'):
            classifier.grow_embedding_layers(dic,
                                             self.unigram_embed_model,
                                             train=train)
            classifier.grow_inference_layers(dic)

    def init_hyperparameters(self):
        if self.unigram_embed_model:
            pretrained_unigram_embed_size = self.unigram_embed_model.wv.syn0[
                0].shape[0]
        else:
            pretrained_unigram_embed_size = 0

        self.hparams = {
            'pretrained_unigram_embed_size': pretrained_unigram_embed_size,
            'pretrained_embed_usage': self.args.pretrained_embed_usage,
            'unigram_embed_size': self.args.unigram_embed_size,
            'rnn_unit_type': self.args.rnn_unit_type,
            'rnn_bidirection': self.args.rnn_bidirection,
            'rnn_batch_first': self.args.rnn_batch_first,
            'rnn_n_layers': self.args.rnn_n_layers,
            'rnn_hidden_size': self.args.rnn_hidden_size,
            'mlp_n_layers': self.args.mlp_n_layers,
            'mlp_hidden_size': self.args.mlp_hidden_size,
            'inference_layer': self.args.inference_layer,
            'embed_dropout': self.args.embed_dropout,
            'rnn_dropout': self.args.rnn_dropout,
            'mlp_dropout': self.args.mlp_dropout,
            'task': self.args.task,
            'lowercasing': self.args.lowercasing,
            'normalize_digits': self.args.normalize_digits,
            'token_freq_threshold': self.args.token_freq_threshold,
            'token_max_vocab_size': self.args.token_max_vocab_size,
        }

        self.log('Init hyperparameters')
        self.log('# arguments')
        for k, v in self.args.__dict__.items():
            message = '{}={}'.format(k, v)
            self.log('# {}'.format(message))
            self.report('[INFO] arg: {}'.format(message))
        self.log('')

    def load_hyperparameters(self, hparams_path):
        hparams = {}
        with open(hparams_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue

                kv = line.split('=')
                key = kv[0]
                val = kv[1]

                if (key == 'pretrained_unigram_embed_size'
                        or key == 'unigram_embed_size' or key == 'rnn_n_layers'
                        or key == 'rnn_hidden_size' or key == 'mlp_n_layers'
                        or key == 'mlp_hidden_size'
                        or key == 'token_freq_threshold'
                        or key == 'token_max_vocab_size'):
                    val = int(val)

                elif (key == 'embed_dropout' or key == 'rnn_dropout'
                      or key == 'mlp_dropout'):
                    val = float(val)

                elif (key == 'rnn_batch_first' or key == 'rnn_bidirection'
                      or key == 'lowercasing' or key == 'normalize_digits'):
                    val = (val.lower() == 'true')

                hparams[key] = val

        self.hparams = hparams
        self.task = self.hparams['task']

        if (self.args.execute_mode != 'interactive'
                and self.hparams['pretrained_unigram_embed_size'] > 0
                and not self.unigram_embed_model):
            self.log('Error: unigram embedding model is necessary.')
            sys.exit()

        if self.unigram_embed_model:
            pretrained_unigram_embed_size = self.unigram_embed_model.wv.syn0[
                0].shape[0]
            if hparams[
                    'pretrained_unigram_embed_size'] != pretrained_unigram_embed_size:
                self.log(
                    'Error: pretrained_unigram_embed_size and size (dimension) of loaded embedding model are conflicted.'
                    .format(hparams['pretrained_unigram_embed_size'],
                            pretrained_unigram_embed_size))
                sys.exit()

    def setup_data_loader(self):
        if self.task == constants.TASK_SEG:
            self.data_loader = segmentation_data_loader.SegmentationDataLoader(
                unigram_vocab=(self.unigram_embed_model.wv
                               if self.unigram_embed_model else set()), )

    def setup_classifier(self):
        dic = self.dic
        hparams = self.hparams

        n_vocab = len(dic.tables['unigram'])
        unigram_embed_size = hparams['unigram_embed_size']

        if 'pretrained_unigram_embed_size' in hparams and hparams[
                'pretrained_unigram_embed_size'] > 0:
            pretrained_unigram_embed_size = hparams[
                'pretrained_unigram_embed_size']
        else:
            pretrained_unigram_embed_size = 0

        if 'pretrained_embed_usage' in hparams:
            pretrained_embed_usage = models.util.ModelUsage.get_instance(
                hparams['pretrained_embed_usage'])
        else:
            pretrained_embed_usage = models.util.ModelUsage.NONE

        if common.is_segmentation_task(self.task):
            n_label = len(dic.tables[constants.SEG_LABEL])
            n_labels = [n_label]
        else:
            self.log('Error: invaliad task {}'.format(self.task))
            sys.exit()

        if (pretrained_embed_usage == models.util.ModelUsage.ADD
                or pretrained_embed_usage == models.util.ModelUsage.INIT):
            if pretrained_unigram_embed_size > 0 and pretrained_unigram_embed_size != unigram_embed_size:
                print(
                    'Error: pre-trained and random initialized unigram embedding vectors must be the same size (dimension) for {} operation: d1={}, d2={}'
                    .format(hparams['pretrained_embed_usage'],
                            pretrained_unigram_embed_size, unigram_embed_size),
                    file=sys.stderr)
                sys.exit()

        predictor = models.tagger.construct_RNNTagger(
            n_vocab=n_vocab,
            unigram_embed_size=unigram_embed_size,
            rnn_unit_type=hparams['rnn_unit_type'],
            rnn_bidirection=hparams['rnn_bidirection'],
            rnn_batch_first=hparams['rnn_batch_first'],
            rnn_n_layers=hparams['rnn_n_layers'],
            rnn_hidden_size=hparams['rnn_hidden_size'],
            mlp_n_layers=hparams['mlp_n_layers'],
            mlp_hidden_size=hparams['mlp_hidden_size'],
            n_labels=n_labels[0],
            use_crf=hparams['inference_layer'] == 'crf',
            # crf_top_k=hparams['crf_top_k'],
            rnn_dropout=hparams['rnn_dropout'],
            embed_dropout=hparams['embed_dropout']
            if 'embed_dropout' in hparams else 0.0,
            mlp_dropout=hparams['mlp_dropout'],
            pretrained_unigram_embed_size=pretrained_unigram_embed_size,
            pretrained_embed_usage=pretrained_embed_usage,
        )

        self.classifier = classifiers.sequence_tagger.SequenceTagger(
            predictor, task=self.task)

    def setup_evaluator(self, evaluator=None):
        evaluator1 = None
        if self.task == constants.TASK_SEG:
            if self.args.evaluation_method == 'normal':
                evaluator1 = FMeasureEvaluator(
                    self.dic.tables[constants.SEG_LABEL].id2str)
        if not evaluator:
            self.evaluator = evaluator1
        else:
            evaluator = evaluator1
