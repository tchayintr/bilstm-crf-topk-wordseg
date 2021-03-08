# A Character-based BiLSTM-CRF Word Segmentation with Top _k_ Segmentation

A character-based word segmentation that employs a character sequence.

### Architecture
- BiLSTM-CRF
- Character-based word segmentation
- CRF as an inference layer
- CoNLL tagging scheme based

### Performance (f1 average score) based on top 1 segmentation
- BIES:
    - micro-avg: 98.46
    - macro-avg: 87.04
- Boundary:
    - micro-avg: 99.18
    - macro-avg: 95.95

### Datasets (based on BEST2010 corpus)
##### See [best2010_cooker](https://github.com/tchayintr/best2010_cooker) for more details.
- train set
  - raw: [best2010_80.shuf.raw.sl](https://resources.aiat.or.th/best2010/thwcc-attn/best2010_80.shuf.raw.sl)
  - segmented: [best2010_80.shuf.seg.sl](https://resources.aiat.or.th/best2010/thwcc-attn/best2010_80.shuf.seg.sl)
- validation set 
  - raw: [best2010_10v.shuf.raw.sl](https://resources.aiat.or.th/best2010/thwcc-attn/best2010_10v.shuf.raw.sl)
  - segmented: [best2010_10v.shuf.seg.sl](https://resources.aiat.or.th/best2010/thwcc-attn/best2010_10v.shuf.seg.sl)
- test set
  - raw: [best2010_10t.shuf.raw.sl](https://resources.aiat.or.th/best2010/thwcc-attn/best2010_10t.shuf.raw.sl)
  - segmented: [best2010_10t.shuf.seg.sl](https://resources.aiat.or.th/best2010/thwcc-attn/best2010_10t.shuf.seg.sl)

### Requirements
- python3 >= 3.7.3
- torch >= 1.7.1+cu101 
- allennlp >= 2.0.1
- numpy >= 1.20.0
- pathlib >= 1.0.1
- gensim >= 3.8.3
- pickle

### Modes
- **train**: training a model
- **eval**: evaluate a model based on [conlleval]( https://github.com/spyysalo/conlleval.py)
- **decode**: decode an input file (unsegmented text) to a segmented words
- **interactive**: decode input text (unsegmented text) in Terminal-like (command prompt)

### Usage
_Modes can be specified in the sample scripts_
#### Training models
- sample_scripts/sample_seg_w.sh
#### Decoding text with top _k_ segmentations
- Specify `CRF_TOP_K` to determine candidate segmentation results

```
usage: segmenter.py [-h] [--execute_mode {train,eval,decode,interactive}]
                    [--task TASK] [--quiet] [--gpu GPU]
                    [--epoch_begin EPOCH_BEGIN] [--epoch_end EPOCH_END]
                    [--break_point BREAK_POINT] [--batch_size BATCH_SIZE]
                    [--grad_clip GRAD_CLIP] [--optimizer OPTIMIZER]
                    [--adam_alpha ADAM_ALPHA] [--adam_beta1 ADAM_BETA1]
                    [--adam_beta2 ADAM_BETA2]
                    [--adam_weight_decay ADAM_WEIGHT_DECAY]
                    [--adagrad_lr ADAGRAD_LR]
                    [--adagrad_lr_decay ADAGRAD_LR_DECAY]
                    [--adagrad_weight_decay ADAGRAD_WEIGHT_DECAY]
                    [--adadelta_lr ADADELTA_LR] [--adadelta_rho ADADELTA_RHO]
                    [--adadelta_weight_decay ADADELTA_WEIGHT_DECAY]
                    [--rmsprop_lr RMSPROP_LR] [--rmsprop_alpha RMSPROP_ALPHA]
                    [--rmsprop_weight_decay RMSPROP_WEIGHT_DECAY]
                    [--sgd_lr SGD_LR] [--sgd_momentum SGD_MOMENTUM]
                    [--sgd_weight_decay SGD_WEIGHT_DECAY]
                    [--scheduler SCHEDULER]
                    [--exponential_gamma EXPONENTIAL_GAMMA]
                    [--model_path MODEL_PATH]
                    [--input_data_path_prefix PATH_PREFIX]
                    [--train_data TRAIN_DATA] [--valid_data VALID_DATA]
                    [--test_data TEST_DATA] [--decode_data DECODE_DATA]
                    [--output_data OUTPUT_DATA] [--input_data_format {sl,wl}]
                    [--output_data_format {sl,wl}]
                    [--output_token_delimiter OUTPUT_TOKEN_DELIM]
                    [--lowercase_alphabets] [--normalize_digits]
                    [--ignored_labels IGNORED_LABELS]
                    [--token_freq_threshold TOKEN_FREQ_THRESHOLD]
                    [--token_max_vocab_size TOKEN_MAX_VOCAB_SIZE]
                    [--pretrained_embed_usage PRETRAINED_EMBED_USAGE]
                    [--unigram_embed_model_path UNIGRAM_EMBED_MODEL_PATH]
                    [--external_dic_path EXTERNAL_DIC_PATH]
                    [--tagging_unit TAGGING_UNIT]
                    [--embed_dropout EMBED_DROPOUT]
                    [--rnn_dropout RNN_DROPOUT]
                    [--rnn_unit_type RNN_UNIT_TYPE] [--rnn_bidirection]
                    [--rnn_batch_first] [--rnn_n_layers RNN_N_LAYERS]
                    [--rnn_hidden_size RNN_HIDDEN_SIZE]
                    [--evaluation_method EVALUATION_METHOD]
                    [--mlp_dropout MLP_DROPOUT]
                    [--unigram_embed_size UNIGRAM_EMBED_SIZE]
                    [--mlp_n_layers MLP_N_LAYERS]
                    [--mlp_hidden_size MLP_HIDDEN_SIZE]
                    [--mlp_activation MLP_ACTIVATION]
                    [--inference_layer_type INFERENCE_LAYER]
                    [--crf_top_k CRF_TOP_K]

optional arguments:
  -h, --help            show this help message and exit
  --execute_mode {train,eval,decode,interactive}, -x {train,eval,decode,interactive}
                        Choose a mode from among 'train', 'eval', 'decode',
                        and 'interactive'
  --task TASK, -t TASK  Select a task
  --quiet, -q           Do not output log file and serialized model file
  --gpu GPU, -g GPU     GPU device id (use CPU if specify a negative value)
  --epoch_begin EPOCH_BEGIN
                        Conduct training from i-th epoch (Default: 1)
  --epoch_end EPOCH_END, -e EPOCH_END
                        Conduct training up to i-th epoch (Default: 5)
  --break_point BREAK_POINT
                        The number of instances which trained model is
                        evaluated and saved (Default: 10000)
  --batch_size BATCH_SIZE, -b BATCH_SIZE
                        The number of examples in each mini-batch (Default:
                        64)
  --grad_clip GRAD_CLIP
                        Gradient norm threshold to clip (Default: 5.0)
  --optimizer OPTIMIZER, -o OPTIMIZER
                        Choose optimizing algorithm from among 'adam',
                        'adedelta', 'adagrad', 'rmsprop', and 'sgd' (Default:
                        adam)
  --adam_alpha ADAM_ALPHA
                        alpha (learning rate) for Adam (Default: 0.001)
  --adam_beta1 ADAM_BETA1
                        beta1 for Adam (Default: 0.9)
  --adam_beta2 ADAM_BETA2
                        beta2 for Adam (Default: 0.999)
  --adam_weight_decay ADAM_WEIGHT_DECAY
                        Weight decay (L2 penalty) for Adam (Default: 0)
  --adagrad_lr ADAGRAD_LR
                        Initial learning rate for AdaGrad (Default: 0.1)
  --adagrad_lr_decay ADAGRAD_LR_DECAY
                        Learning rate decay for AdaGrad (Default: 0)
  --adagrad_weight_decay ADAGRAD_WEIGHT_DECAY
                        Weight decay (L2 penalty) for AdaGrad (Default: 0)
  --adadelta_lr ADADELTA_LR
                        Initial learning rate for AdaDelta (Default: 1.0)
  --adadelta_rho ADADELTA_RHO
                        rho for AdaDelta (Default: 0.9)
  --adadelta_weight_decay ADADELTA_WEIGHT_DECAY
                        Weight decay (L2 penalty) for AdaDelta (Default: 0)
  --rmsprop_lr RMSPROP_LR
                        Initial learning rate for RMSprop (Default: 0.1)
  --rmsprop_alpha RMSPROP_ALPHA
                        alpha for RMSprop (Default: 0.99)
  --rmsprop_weight_decay RMSPROP_WEIGHT_DECAY
                        Weight decay (L2 penalty) for RMSprop (Default: 0)
  --sgd_lr SGD_LR       Initial learning rate for SGD optimizers (Default:
                        0.1)
  --sgd_momentum SGD_MOMENTUM
                        Momentum factor for SGD (Default: 0.0)
  --sgd_weight_decay SGD_WEIGHT_DECAY
                        Weight decay (L2 penalty) for SGD (Default: 0)
  --scheduler SCHEDULER, -s SCHEDULER
                        Choose scheduler for optimizer from among
                        'exponential' (Default: exponential)
  --exponential_gamma EXPONENTIAL_GAMMA
                        Multiplicative factor of learning rate decay (Learning
                        rate decay) for Exponential (Default: 0.99)
  --model_path MODEL_PATH, -m MODEL_PATH
                        pt/pkl file path of the trained model. 'xxx.hyp' and
                        'xxx.s2i' files are also read simultaneously when you
                        specify 'xxx_izzz.pt/pkl' file
  --input_data_path_prefix PATH_PREFIX, -p PATH_PREFIX
                        Path prefix of input data
  --train_data TRAIN_DATA
                        File path succeeding 'input_data_path_prefix' of
                        training data
  --valid_data VALID_DATA
                        File path succeeding 'input_data_path_prefix' of
                        validation data
  --test_data TEST_DATA
                        File path succeeding 'input_data_path_prefix' of test
                        data
  --decode_data DECODE_DATA
                        File path of input text which succeeds
                        'input_data_path_prefix'
  --output_data OUTPUT_DATA
                        File path to output parsed text
  --input_data_format {sl,wl}, -f {sl,wl}
                        Choose format of input data from among 'sl' and 'wl'
                        (Default: sl)
  --output_data_format {sl,wl}
                        Choose format of output data from among 'sl' and 'wl'
                        (Default: sl)
  --output_token_delimiter OUTPUT_TOKEN_DELIM
                        Specify delimiter symbol between words for SL format
                        when output analysis results on decode/interactive
                        mode (Default ' ')
  --lowercase_alphabets
                        Lowercase alphabets in input text
  --normalize_digits    Normalize digits by the same symbol in input text
  --ignored_labels IGNORED_LABELS
                        Sepecify labels to be ignored on evaluation by format
                        'label_1,label_2,...,label_N'
  --token_freq_threshold TOKEN_FREQ_THRESHOLD
                        Token frequency threshold. Tokens whose frequency are
                        lower than the the threshold are regarded as unknown
                        tokens (Default: 1)
  --token_max_vocab_size TOKEN_MAX_VOCAB_SIZE
                        Maximum size of token vocaburaly. low frequency tokens
                        are regarded as unknown tokens so that vocaburaly size
                        does not exceed the specified size so much if set
                        positive value (Default: -1)
  --pretrained_embed_usage PRETRAINED_EMBED_USAGE
                        Specify usage of pretrained embedding model from among
                        'init' 'concat' and 'add'
  --unigram_embed_model_path UNIGRAM_EMBED_MODEL_PATH
                        File path of pretrained model of token (character or
                        word) unigram embedding
  --external_dic_path EXTERNAL_DIC_PATH
                        File path of external word dictionary listing known
                        words
  --tagging_unit TAGGING_UNIT
                        Specify tagging unit, 'single' for character-based
                        model
  --embed_dropout EMBED_DROPOUT
                        Dropout ratio for embedding layers (Default: 0.0)
  --rnn_dropout RNN_DROPOUT
                        Dropout ratio for RNN vertical layers (Default: 0.2)
  --rnn_unit_type RNN_UNIT_TYPE
                        Choose unit type of RNN from among 'lstm', 'gru' and
                        'plain (tanh) ' (Default: lstm)
  --rnn_bidirection     Use bidirectional RNN (Default: False)
  --rnn_batch_first     To provide the input and output tensor as (batch, seq,
                        feature) (Default: False)
  --rnn_n_layers RNN_N_LAYERS
                        The number of RNN layers (Default: 1)
  --rnn_hidden_size RNN_HIDDEN_SIZE
                        The size of hidden units (dimension) for RNN (Default:
                        256)
  --evaluation_method EVALUATION_METHOD
                        Evaluation method for segmentation
  --mlp_dropout MLP_DROPOUT
                        Dropout ratio for MLP of sequence labeling model
                        (Default: 0.0)
  --unigram_embed_size UNIGRAM_EMBED_SIZE
                        The size (dimension) of token (character or word)
                        unigram embedding (Default: 128)
  --mlp_n_layers MLP_N_LAYERS
                        The number of layers of MLP of sequence labeling
                        model. The last layer projects input hidden vector to
                        dimensional space of number of labels (Default: 1)
  --mlp_hidden_size MLP_HIDDEN_SIZE
                        The size of hidden units (dimension) of MLP of
                        sequence labeling model (Default: 300)
  --mlp_activation MLP_ACTIVATION
                        Choose type of activation function for Muti-layer
                        perceptron from between'sigmoid' and 'relu' (Default:
                        relu)
  --inference_layer_type INFERENCE_LAYER
                        Choose type of inference layer for sequence labeling
                        model from between 'softmax' and 'crf' (Default: crf)
  --crf_top_k CRF_TOP_K
                        Specify the top k for CRF (viterbi algorithm)
                        (Default: 5)
```

#### Data format
- **sl**: sentence line
- **wl**: word line

### Contributions
- **[Okumura-Takamura-Funakoshi Lab](http://lr-www.pi.titech.ac.jp)**: Natural Language Processing Group at Tokyo Institute of Technology
- **[AIAT](https://aiat.or.th)**: Artificial Intelligence Association of Thailand

### Acknowledgement
-  Implementations based on modification of [seikanlp](https://github.com/shigashiyama/seikanlp)

### Citation
- N/A
