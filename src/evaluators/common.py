import sys

import constants
from tools import conlleval
from tools.conlleval import FCounts


# Counts
class Counts(object):
    def merge(self, counts):
        # to be implemented in sub-class
        pass


class ACounts(Counts):
    def __init__(self):
        self.total = 0
        self.correct = 0

    def merge(self, counts):
        if not isinstance(counts, ACounts):
            print('Invalid count object', file=sys.stderr)
            return

        self.total += counts.total
        self.correct += counts.correct


class FACounts(Counts):
    def __init__(self):
        self.l1 = FCounts()
        self.l2 = ACounts()

    def merge(self, counts):
        if not isinstance(counts, FACounts):
            print('Invalid count object', file=sys.stderr)
            return

        self.l1.merge(counts.l1)
        self.l2.merge(counts.l2)


# Calculators
class AccuracyCalculator(object):
    def __init__(self, ignore_head=False, ignored_labels=set()):
        self.ignore_head = ignore_head
        self.ignored_labels = ignored_labels

    def __call__(self, ts, ys):
        counts = ACounts()
        for t, y in zip(ts, ys):
            if self.ignore_head:
                t = t[1:]
                y = y[1:]

            for ti, yi in zip(t, y):
                if int(ti) in self.ignored_labels:
                    continue

                counts.total += 1
                if ti == yi:
                    counts.correct += 1

        return counts


class FMeasureCalculator(object):
    def __init__(self, id2label):
        self.id2label = id2label

    def __call__(self, xs, ts, ys):
        counts = FCounts()
        for x, t, y in zip(xs, ts, ys):
            generator = self.generate_lines(x, t, y)
            counts = conlleval.evaluate(generator, counts=counts)

        return counts

    def generate_lines(self, x, t, y):
        i = 0
        while True:
            if i == len(x):
                break
            elif int(x[i]) == constants.PADDING_LABEL:
                break

            x_str = str(x[i])
            t_str = self.id2label[int(t[i])] if int(t[i]) > -1 else 'NONE'
            y_str = self.id2label[int(y[i])] if int(y[i]) > -1 else 'NONE'

            yield [x_str, t_str, y_str]
            i += 1


class FMeasureAndAccuracyCalculator(object):
    def __init__(self, id2label):
        self.id2label = id2label

    def __call__(self, xs, t1s, t2s, y1s, y2s):
        counts = FACounts()
        if not t2s or not y2s:
            t2s = [None] * len(xs)
            y2s = [None] * len(xs)

        for x, t1, t2, y1, y2 in zip(xs, t1s, t2s, y1s, y2s):
            generator_seg = self.generate_lines_seg(x, t1, y1)
            counts.l1 = conlleval.evaluate(generator_seg, counts=counts.l1)

            if t2 is not None:
                for ti, yi in zip(t2, y2):
                    counts.l2.total += 1
                    if ti == yi:
                        counts.l2.correct += 1

        return counts

    def generate_lines_seg(self, x, t, y):
        i = 0
        while True:
            if i == len(x):
                break
            x_str = str(x[i])
            t_str = self.id2label[int(t[i])] if int(t[i]) > -1 else 'NONE'
            y_str = self.id2label[int(y[i])]
            tseg_str = t_str.split('-')[0]
            yseg_str = y_str.split('-')[0]

            yield [x_str, tseg_str, yseg_str]
            i += 1


# Evaluators
class FMeasureEvaluator(object):
    def __init__(self, id2label):
        self.calculator = FMeasureCalculator(id2label)

    def calculate(self, *inputs):
        counts = self.calculator(*inputs)
        return counts

    def report_results(self, sen_counter, counts, loss, file=sys.stderr):
        ave_loss = loss / counts.token_counter
        met = conlleval.calculate_metrics(counts.correct_chunk,
                                          counts.found_guessed,
                                          counts.found_correct,
                                          counts.correct_tags,
                                          counts.token_counter)

        print('ave loss: %.5f' % ave_loss, file=file)
        print('sen, token, chunk, chunk_pred: {} {} {} {}'.format(
            sen_counter, counts.token_counter, counts.found_correct,
            counts.found_guessed),
              file=file)
        print('TP, FP, FN: %d %d %d' % (met.tp, met.fp, met.fn), file=file)
        print('A, P, R, F:%6.2f %6.2f %6.2f %6.2f' %
              (100. * met.acc, 100. * met.prec, 100. * met.rec,
               100. * met.fscore),
              file=file)

        res = '%.2f\t%.2f\t%.2f\t%.2f\t%.4f' % ((100. * met.acc),
                                                (100. * met.prec),
                                                (100. * met.rec),
                                                (100. * met.fscore), ave_loss)
        return res


class AccuracyEvaluator(object):
    def __init__(self, ignore_head=False, ignored_labels=set()):
        self.calculator = AccuracyCalculator(ignore_head, ignored_labels)

    def calculate(self, *inputs):
        ts = inputs[1]
        ys = inputs[2]
        counts = self.calculator(ts, ys)
        return counts

    def report_results(self, sen_counter, counts, loss=None, file=sys.stderr):
        ave_loss = loss / counts.total if loss is not None else None
        acc = 1. * counts.correct / counts.total

        if ave_loss is not None:
            print('ave loss: %.5f' % ave_loss, file=file)
        print('sen, token, correct: {} {} {}'.format(sen_counter, counts.total,
                                                     counts.correct),
              file=file)
        print('A:%6.2f' % (100. * acc), file=file)

        if ave_loss is not None:
            res = '%.2f\t%.4f' % ((100. * acc), ave_loss)
        else:
            res = '%.2f' % (100. * acc)

        return res
