#!/usr/bin/env bash

BATCH_SIZE=32
MODEL=CNN_AVE
MAX_HOPS=5
DEVICE=6
VERSION=1
N_EPOCHS=50
QUERY_TYPE=RELATION
DECAY_RATIO=0.5
#lr=0.001
#lr=0.00125
#lr=0.00075
#lr=5e-4
lr=0.01
DATA_ROOT=/data/yanjianhao/nlp/torch/torch_NRE/data

CUDA_VISIBLE_DEVICES=$DEVICE python trainer.py --model=$MODEL \
                                                --dataset_dir=$DATA_ROOT \
                                                --cuda \
                                                --order_embed \
                                                --position_embedding \
                                                --max_hops=$MAX_HOPS \
                                                --batch_size=$BATCH_SIZE \
                                                --mem_version=$VERSION \
                                                --max_epochs=$N_EPOCHS \
                                                --remove_origin_query \
                                                --query_type=$QUERY_TYPE \
                                                --problem=WIKI-TIME \
                                                --lr=$lr \
                                                --decay_ratio=$DECAY_RATIO \
                                                --use_noise_and_clip
