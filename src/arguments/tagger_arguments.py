from pathlib import Path
import sys

from arguments.arguments import ArgumentLoader
import constants


class TaggerArgumentLoader(ArgumentLoader):
    def parse_args(self):
        return super().parse_args()

    def get_full_parser(self):
        parser = super().get_full_parser()

        # data paths and related options
        parser.add_argument(
            '--unigram_embed_model_path',
            type=Path,
            default=None,
            help=
            'File path of pretrained model of token (character or word) unigram embedding'
        )
        parser.add_argument(
            '--external_dic_path',
            type=Path,
            default=None,
            help='File path of external word dictionary listing known words')

        # options for data pre/post-processing
        ## common
        ### sequence labeling
        parser.add_argument(
            '--tagging_unit',
            default='single',
            help='Specify tagging unit, \'single\' for character-based model')

        # model parameters
        ## options for model architecture and parameters
        ### common
        parser.add_argument(
            '--embed_dropout',
            type=float,
            default=0.0,
            help='Dropout ratio for embedding layers (Default: 0.0)')
        parser.add_argument(
            '--rnn_dropout',
            type=float,
            default=0.2,
            help='Dropout ratio for RNN vertical layers (Default: 0.2)')
        parser.add_argument(
            '--rnn_unit_type',
            default='lstm',
            help=
            'Choose unit type of RNN from among \'lstm\', \'gru\' and \'plain (tanh) \' (Default: lstm)'
        )
        parser.add_argument('--rnn_bidirection',
                            action='store_true',
                            help='Use bidirectional RNN (Default: False)')
        parser.add_argument(
            '--rnn_batch_first',
            action='store_true',
            help=
            'To provide the input and output tensor as (batch, seq, feature) (Default: False)'
        )
        parser.add_argument('--rnn_n_layers',
                            type=int,
                            default=1,
                            help='The number of RNN layers (Default: 1)')
        parser.add_argument(
            '--rnn_hidden_size',
            type=int,
            default=256,
            help='The size of hidden units (dimension) for RNN (Default: 256)')

        ### segmentation evaluation
        parser.add_argument('--evaluation_method',
                            default='normal',
                            help='Evaluation method for segmentation')

        ### segmentation
        parser.add_argument(
            '--mlp_dropout',
            type=float,
            default=0.0,
            help=
            'Dropout ratio for MLP of sequence labeling model (Default: 0.0)')
        parser.add_argument(
            '--unigram_embed_size',
            type=int,
            default=128,
            help=
            'The size (dimension) of token (character or word) unigram embedding (Default: 128)'
        )
        parser.add_argument(
            '--mlp_n_layers',
            type=int,
            default=1,
            help=
            'The number of layers of MLP of sequence labeling model. The last layer projects input hidden vector to dimensional space of number of labels (Default: 1)'
        )
        parser.add_argument(
            '--mlp_hidden_size',
            type=int,
            default=300,
            help=
            'The size of hidden units (dimension) of MLP of sequence labeling model (Default: 300)'
        )
        parser.add_argument(
            '--mlp_activation',
            default='relu',
            help=
            'Choose type of activation function for Muti-layer perceptron from between'
            + '\'sigmoid\' and \'relu\' (Default: relu)')
        parser.add_argument(
            '--inference_layer_type',
            dest='inference_layer',
            default='crf',
            help=
            'Choose type of inference layer for sequence labeling model from between '
            + '\'softmax\' and \'crf\' (Default: crf)')
        parser.add_argument(
            '--crf_top_k',
            type=int,
            default=1,
            help='Specify the top k for CRF (viterbi algorithm) (Default: 5)')

        return parser

    def get_minimum_parser(self, args):
        parser = super().get_minimum_parser(args)
        parser.add_argument('--evaluation_method',
                            default=args.evaluation_method)
        if not (args.evaluation_method == 'normal'):
            print('Error: evaluation_method must be specified to \'normal\''.
                  format(args.evaluation_method),
                  file=sys.stderr)
            sys.exit()

        parser.add_argument('--unigram_embed_model_path',
                            type=Path,
                            default=args.unigram_embed_model_path)
        parser.add_argument('--embed_dropout',
                            type=float,
                            default=args.embed_dropout)
        parser.add_argument('--rnn_dropout',
                            type=float,
                            default=args.rnn_dropout)

        # specific options for segmentation/tagging
        parser.add_argument('--external_dic_path',
                            type=Path,
                            default=args.external_dic_path)
        parser.add_argument('--mlp_dropout',
                            type=float,
                            default=args.mlp_dropout)
        parser.add_argument('--tagging_unit', default=args.tagging_unit)

        # specific options for hybrid segmentation
        if args.tagging_unit == 'hybrid':
            parser.add_argument('--lattice_n_paths',
                                type=int,
                                default=args.lattice_n_paths)
            parser.add_argument('--biaffine_dropout',
                                type=float,
                                default=args.biaffine_dropout)
            parser.add_argument('--chunk_vector_dropout',
                                type=float,
                                default=args.chunk_vector_dropout)
            parser.add_argument('--chunk_embed_model_path',
                                default=args.chunk_embed_model_path)
            parser.add_argument('--gen_oov_chunk_for_test',
                                action='store_const',
                                const=False,
                                default=False)

        if args.execute_mode == 'train':
            if (not args.model_path and args.unigram_embed_model_path):
                if not (args.pretrained_embed_usage == 'init'
                        or args.pretrained_embed_usage == 'concat'
                        or args.pretrained_embed_usage == 'add'):
                    print(
                        'Error: pretrained_embed_usage must be specified among from {init, concat, add}:{}'
                        .format(args.pretrained_embed_usage),
                        file=sys.stderr)
                    sys.exit()

                if args.unigram_embed_model_path and args.unigram_embed_size <= 0:
                    print(
                        'Error: unigram_embed_size must be positive value to use pretrained unigram embed model: {}'
                        .format(args.unigram_embed_size),
                        file=sys.stderr)
                    sys.exit()

            parser.add_argument('--unigram_embed_size',
                                type=int,
                                default=args.unigram_embed_size)
            parser.add_argument('--rnn_unit_type', default=args.rnn_unit_type)
            parser.add_argument('--rnn_bidirection',
                                action='store_true',
                                default=args.rnn_bidirection)
            parser.add_argument('--rnn_batch_first',
                                action='store_true',
                                default=args.rnn_batch_first)
            parser.add_argument('--rnn_n_layers',
                                type=int,
                                default=args.rnn_n_layers)
            parser.add_argument('--rnn_hidden_size',
                                type=int,
                                default=args.rnn_hidden_size)

            # specific options for segmentation
            parser.add_argument('--mlp_n_layers',
                                type=int,
                                default=args.mlp_n_layers)
            parser.add_argument('--mlp_hidden_size',
                                type=int,
                                default=args.mlp_hidden_size)
            parser.add_argument('--mlp_activation',
                                default=args.mlp_activation)
            parser.add_argument('--inference_layer_type',
                                dest='inference_layer',
                                default=args.inference_layer)

        if args.execute_mode == 'decode' or args.execute_mode == 'interactive':
            parser.add_argument('--crf_top_k',
                                type=int,
                                default=args.crf_top_k)

        return parser

    def add_input_data_format_option(self, parser, args):
        if args.execute_mode == 'interactive':
            pass

        elif args.task == constants.TASK_SEG:
            if args.input_data_format == 'sl':
                parser.add_argument('--input_data_format',
                                    '-f',
                                    default=args.input_data_format)
            elif args.input_data_format == 'wl':
                if args.execute_mode == 'train' or args.execute_mode == 'eval':
                    parser.add_argument('--input_data_format',
                                        '-f',
                                        default=args.input_data_format)
                elif args.execute_mode == 'decode':
                    print(
                        'Error: input data format for task={}/mode={} must be specified as \'sl\'. Input: {}'
                        .format(args.task, args.execute_mode,
                                args.input_data_format),
                        file=sys.stderr)
                    sys.exit()
            else:
                if args.execute_mode == 'decode':
                    parser.add_argument('--input_data_format',
                                        '-f',
                                        default='sl')
                else:
                    print(
                        'Error: input data format for task={}/mode={} must be specified among from {sl, wl}. Input: {}'
                        .format(args.task, args.execute_mode,
                                args.input_data_format),
                        file=sys.stderr)
                    sys.exit()
