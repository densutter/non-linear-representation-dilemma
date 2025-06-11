#!/bin/bash

set -x

MODEL="EleutherAI/pythia-410m"
DTYPE="float32"
HIDDEN_SIZE=1024
LR=0.0005
EPOCHS=2
REVNET_HIDDEN=64

DEFAULT_ARGS="--dtype $DTYPE --model_name $MODEL --revnet_depth 1 --in_features $HIDDEN_SIZE --model_init_mode 0 --layer_index 12 --lr $LR --lr_warmup_steps 500 --early_stopping_epochs 100 --max_epochs $EPOCHS --revnet_hidden_size $REVNET_HIDDEN"
CMD="python scripts/das_llm.py $DEFAULT_ARGS"

for STEP in step0 main; do
    $CMD --transformation_type "Rotation" --name-split --model_revision $STEP 
    
    for BLOCKS in 8 16; do
        echo "Running with learning rate: $LR"
        $CMD  --revnet_blocks $BLOCKS --model_revision $STEP --name-split
    done
done