## Common parameters
set -e

GPU_ID=1
TASK=seg
# BATCH=50
BATCH=128


################################################
# Train a segmentation model

MODE=train
EP_BEGIN=1
EP_END=20
# BREAK_POINT=100
BREAK_POINT=65536
CHAR_EMB_SIZE=128
RNN_LAYERS=2
RNN_HIDDEN_SIZE=600
MLP_LAYERS=2
# TRAIN_DATA=data/samples/best2010_sample_100.seg.sl
TRAIN_DATA=data/best2010/v1/best2010_80.shuf.seg.sl
# VALID_DATA=data/samples/best2010_sample_20.seg.sl
VALID_DATA=data/best2010/v1/best2010_10v.shuf.seg.sl
RNN_DROPOUT=0.4
INFERENCE_LAYER=crf
INPUT_FORMAT=sl

# python3 src/segmenter.py \
#        --task $TASK \
#        --execute_mode $MODE \
#        --gpu $GPU_ID \
#        --epoch_begin $EP_BEGIN \
#        --epoch_end $EP_END \
#        --break_point $BREAK_POINT \
#        --batch_size $BATCH \
#        --unigram_embed_size $CHAR_EMB_SIZE \
#        --rnn_bidirection \
#        --rnn_n_layers $RNN_LAYERS \
#        --rnn_hidden_size $RNN_HIDDEN_SIZE \
#        --rnn_dropout $RNN_DROPOUT \
#        --mlp_n_layers $MLP_LAYERS \
#        --inference_layer $INFERENCE_LAYER \
#        --train_data $TRAIN_DATA \
#        --valid_data $VALID_DATA \
#        --input_data_format $INPUT_FORMAT \
#        --rnn_batch_first \
#        --quiet \

# MODEL=models/main/yyyymmdd_hhmm_ex.yyy.pt
# MODEL=models/main/20210207_0014/20210207_0014_e12.184.pt
MODEL=models/main/20200710_1611/20200710_1611_e17.723.pt


################################################
# Evaluate the learned model

MODE=eval
# TEST_DATA=data/samples/best2010_sample_10.raw.sl
TEST_DATA=data/best2010/v1/best2010_10t.shuf.raw.sl
INPUT_FORMAT=sl

# python3 src/segmenter.py \
#        --task $TASK \
#        --execute_mode $MODE \
#        --gpu $GPU_ID \
#        --batch_size $BATCH \
#        --test_data $TEST_DATA \
#        --input_data_format $INPUT_FORMAT \
#        --model_path $MODEL \
#        --quiet


################################################
# Segment a raw text by the learned model

MODE=decode
# DECODE_DATA=data/samples/best2010_sample_10.raw.sl
DECODE_DATA=data/best2010/v1/best2010_10t.shuf.raw.sl
OUTPUT_DATA=decode_ch_top40.sl
OUTPUT_FORMAT=sl
CRF_TOP_K=40

python3 src/segmenter.py \
       --task $TASK \
       --execute_mode $MODE \
       --gpu $GPU_ID \
       --batch_size $BATCH \
       --decode_data $DECODE_DATA \
       --model_path $MODEL \
       --output_data_format $OUTPUT_FORMAT \
       --output_data $OUTPUT_DATA \
       --crf_top_k $CRF_TOP_K \
       # --quiet \


################################################
# Segment a raw text by the learned model via interactive shell

MODE=interactive
OUTPUT_FORMAT=sl
CRF_TOP_K=3

# python3 src/segmenter.py \
#        --task $TASK \
#        --execute_mode $MODE \
#        --gpu $GPU_ID \
#        --model_path $MODEL \
#        --output_data_format $OUTPUT_FORMAT \
#        --crf_top_k $CRF_TOP_K \
#        --quiet \
