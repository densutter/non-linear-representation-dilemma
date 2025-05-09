#!/bin/bash

set -x

MODEL="EleutherAI/pythia-410m"
DTYPE="float32"
HIDDEN_SIZE=1024
LR=0.0005
EPOCHS=2
REVNET_HIDDEN=64

DEFAULT_ARGS="--dtype $DTYPE --revnet_blocks 8 --revnet_depth 1 --in_features $HIDDEN_SIZE --model_init_mode 0 --layer_index 12 --lr $LR --lr_warmup_steps 500 --early_stopping_epochs 15 --max_epochs $EPOCHS --revnet_hidden_size $REVNET_HIDDEN"
CMD="python scripts/das_llm.py $DEFAULT_ARGS"


for SEED in 1 2 3 4 5; do
    for REVISION in step0 main; do
        $CMD --transformation_type "Rotation" --model_name EleutherAI/pythia-410m-seed$SEED --model_revision $REVISION
        $CMD --transformation_type "RevNet" --model_name EleutherAI/pythia-410m-seed$SEED --model_revision $REVISION
    done
done