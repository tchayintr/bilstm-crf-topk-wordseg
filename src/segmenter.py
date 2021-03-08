import sys

from core import Core
from arguments import tagger_arguments
from trainers import tagger_trainer


class Segmenter(Core):
    def __init__(self):
        super().__init__()

    def get_args(self):
        parser = tagger_arguments.TaggerArgumentLoader()
        args = parser.parse_args()
        return args

    def get_trainer(self, args):
        if args.tagging_unit == 'single':
            trainer = tagger_trainer.TaggerTrainer(args)
        else:
            print('Error: the following argument is invalid for {} mode: {}'.
                  format(args.execute_mode, '--tagging_unit'))
            sys.exit()

        return trainer


if __name__ == '__main__':
    analyzer = Segmenter()
    analyzer.run()
