#!/bin/bash

set -x

MODEL="EleutherAI/pythia-410m"
DTYPE="float32"
HIDDEN_SIZE=1024
REVISION=$1
LR=0.0005
EPOCHS=2
REVNET_HIDDEN=64

DEFAULT_ARGS="--dtype $DTYPE --model_name $MODEL --revnet_depth 1 --in_features $HIDDEN_SIZE --model_init_mode 0 --model_revision $REVISION --layer_index 12 --lr $LR --lr_warmup_steps 500 --early_stopping_epochs 15 --max_epochs $EPOCHS --revnet_hidden_size $REVNET_HIDDEN"
CMD="python scripts/das_llm.py $DEFAULT_ARGS"

$CMD --transformation_type "Rotation"

for BLOCKS in 1 2 4 8 16; do
    echo "Running with learning rate: $LR"
    $CMD  --revnet_blocks $BLOCKS 
done