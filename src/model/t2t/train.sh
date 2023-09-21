#!/bin/bash

ARCH=aarch64

TRAINER=t2t-trainer
DATA_DIR=/root/data/$(ARCH)/csv
OUTPUT_DIR=/root/model/t2t/$(ARCH)

TRAINER \
    --generate_data \
    --data_dir=$(DATA_DIR) \
    --output_dir=$(OUTPUT_DIR) \
    --problem=decompiler\
    --model=transformer \
    --hparams_set=transformer_base \
    --train_steps=1000 \
    --eval_steps=100
