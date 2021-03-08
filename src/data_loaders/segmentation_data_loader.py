import torch
import sys

import constants
from data_loaders import data_loader
from data_loaders.data_loader import Data, DataLoader, RestorableData
import dictionary


class SegmentationDataLoader(DataLoader):
    def __init__(self, unigram_vocab=set()):
        self.unigram_vocab = unigram_vocab

    def load_gold_data(self, path, data_format, dic=None, train=True):
        if data_format == constants.SL_FORMAT:
            data, dic = self.load_gold_data_SL(path, dic, train)
        elif data_format == constants.WL_FORMAT:
            data, dic = self.load_gold_data_WL(path, dic, train)
        return data, dic

    def load_decode_data(self, path, data_format, dic=None):
        return self.load_decode_data_SL(path, dic)

    def parse_commandline_input(self, line, dic):
        get_unigram_id = dic.tables[constants.UNIGRAM].get_id
        org_token_seq = [char for char in line]
        org_token_seqs = [org_token_seq]

        uni_seq = [get_unigram_id(char) for char in line]
        uni_seqs = [uni_seq]

        inputs = [uni_seqs]
        outputs = []
        orgdata = [org_token_seqs]

        return RestorableData(inputs, outputs, orgdata=orgdata)

    def load_gold_data_SL(self, path, dic=None, train=True):
        if not dic:
            dic = init_dictionary()

        get_unigram_id = dic.tables[constants.UNIGRAM].get_id
        get_seg_id = dic.tables[constants.SEG_LABEL].get_id

        token_seqs = []
        seg_seqs = []

        ins_cnt = 0

        with open(path) as f:
            for line in f:
                line = self.normalize_input_line(line)
                if len(line) <= 1:
                    continue
                # elif line[0] == constants.COMMENT_SYM:
                #     continue

                entries = line.split(constants.SL_TOKEN_DELIM)
                uni_seq = []
                seg_seq = []
                raw_sen = ''

                for entry in entries:
                    token = entry
                    tlen = len(token)
                    raw_sen += token

                    uni_seq.extend(
                        [get_unigram_id(token[i], True) for i in range(tlen)])
                    seg_seq.extend([
                        get_seg_id(data_loader.get_label_BIES(i, tlen - 1),
                                   update=train) for i in range(tlen)
                    ])

                token_seqs.append(uni_seq)
                seg_seqs.append(seg_seq)

                ins_cnt += 1
                if ins_cnt % constants.NUM_FOR_REPORTING == 0:
                    print('Read {} sentences'.format(ins_cnt), file=sys.stderr)

        inputs = [token_seqs]
        outputs = [seg_seqs]

        return Data(inputs, outputs), dic

    def load_gold_data_WL(self, path, dic=None, train=True):
        if not dic:
            dic = init_dictionary()

        get_unigram_id = dic.tables[constants.UNIGRAM].get_id
        get_seg_id = dic.tables[constants.SEG_LABEL].get_id

        token_seqs = []
        seg_seqs = []

        ins_cnt = 0

        with open(path) as f:
            uni_seq = []
            seg_seq = []
            sen = ''

            for lnum, line in enumerate(f):
                line = self.normalize_input_line(line)
                if len(line) == 0:
                    if len(uni_seq) > 0:
                        token_seqs.append(uni_seq)

                        uni_seq = []
                        if seg_seq:
                            seg_seqs.append(seg_seq)
                            seg_seq = []
                        sen = ''

                        ins_cnt += 1
                        if ins_cnt % constants.NUM_FOR_REPORTING == 0:
                            print('Read {} sentences'.format(ins_cnt),
                                  file=sys.stderr)

                    continue
                elif line[0] == constants.COMMENT_SYM:
                    continue

                token = line
                tlen = len(token)
                sen += token

                uni_seq.extend(
                    [get_unigram_id(token[i], True) for i in range(tlen)])
                seg_seq.extend([
                    get_seg_id(data_loader.get_label_BIES(i, tlen - 1),
                               update=train) for i in range(tlen)
                ])

            # register last sentence
            if uni_seq:
                token_seqs.append(uni_seq)
                if seg_seq:
                    seg_seqs.append(seg_seq)

        inputs = [token_seqs]
        outputs = [seg_seqs]

        return Data(inputs, outputs), dic

    def load_decode_data_SL(self, path, dic):
        get_unigram_id = dic.tables[constants.UNIGRAM].get_id

        org_token_seqs = []
        token_seqs = []

        ins_cnt = 0
        with open(path) as f:
            for line in f:
                line = self.normalize_input_line(line)
                if len(line) == 0:
                    continue
                # elif line[0] == constants.COMMENT_SYM:
                #     continue

                org_token_seqs.append([char for char in line])
                uni_seq = [get_unigram_id(char) for char in line]

                token_seqs.append(uni_seq)

                ins_cnt += 1
                if ins_cnt % constants.NUM_FOR_REPORTING == 0:
                    print('Read {} sentences'.format(ins_cnt), file=sys.stderr)

        inputs = [token_seqs]
        outputs = []
        orgdata = [org_token_seqs]

        return RestorableData(inputs, outputs, orgdata=orgdata)


def get_char_type(char):
    if len(char) != 1:
        return

    if char == '\u30fc':
        return constants.TYPE_LONG
    elif '\u0e01' <= char <= '\u0e5b':
        return constants.TYPE_THAI
    elif '\u3041' <= char <= '\u3093':
        return constants.TYPE_HIRA
    elif '\u30A1' <= char <= '\u30F4':
        return constants.TYPE_KATA
    elif '\u4e8c' <= char <= '\u9fa5':
        return constants.TYPE_KANJI
    elif '\uff10' <= char <= '\uff19' or '0' <= char <= '9':
        return constants.TYPE_DIGIT
    elif '\uff21' <= char <= '\uff5a' or 'A' <= char <= 'z':
        return constants.TYPE_ALPHA
    elif char == '\u3000' or char == ' ':
        return constants.TYPE_SPACE
    elif '!' <= char <= '~':
        return constants.TYPE_ASCII_OTHER
    else:
        return constants.TYPE_SYMBOL


def get_segmentation_spans(label_seq):
    spans = []
    first = -1
    for i, label in enumerate(label_seq):
        if label == 3:  # 'S'
            spans.append((i, i + 1))
        elif label == 0:  # 'B'
            first = i
        elif label == 2:  # 'E'
            spans.append((first, i + 1))
    return spans


def load_external_dictionary(path, dic=None):
    if not dic:
        dic = init_dictionary()
    get_unigram_id = dic.tables[constants.UNIGRAM].get_id

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(constants.COMMENT_SYM) and len(line) == 1:
                continue
            if len(line) == 0:
                continue

            word = line
            char_ids = [get_unigram_id(char, update=True) for char in word]
    dic.create_id2strs()

    return dic


def init_dictionary():
    dic = dictionary.Dictionary()

    # unigram
    dic.create_table(constants.UNIGRAM)
    dic.tables[constants.UNIGRAM].set_unk(constants.UNK_SYMBOL)

    # segmentation label
    dic.create_table(constants.SEG_LABEL)
    for label in constants.SEG_LABELS:
        dic.tables[constants.SEG_LABEL].get_id(label, update=True)

    return dic
