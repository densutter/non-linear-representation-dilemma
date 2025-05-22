#!/bin/bash

set -x

DTYPE="float64"
LR=0.001
EPOCHS=4
REVNET_HIDDEN=64
BLOCKS=8

for MODEL in "EleutherAI/pythia-70m" "EleutherAI/pythia-31m" "EleutherAI/pythia-160m"; do
	if [[ "$MODEL" == "EleutherAI/pythia-160m" ]]; then
		LAYER_INDEX=6
	else
		LAYER_INDEX=3
	fi

	# Set hidden size based on model if needed; here's a default for example
	case "$MODEL" in
		"EleutherAI/pythia-160m") HIDDEN_SIZE=768 ;;
		"EleutherAI/pythia-70m")  HIDDEN_SIZE=512 ;;
		"EleutherAI/pythia-31m")  HIDDEN_SIZE=256 ;;
		*) HIDDEN_SIZE=1024 ;;
	esac

	DEFAULT_ARGS="--revnet_blocks $BLOCKS --dtype $DTYPE --revnet_depth 1 --in_features $HIDDEN_SIZE --model_init_mode 0 --layer_index $LAYER_INDEX --lr $LR --lr_warmup_steps 500 --early_stopping_epochs 15 --max_epochs $EPOCHS --revnet_hidden_size $REVNET_HIDDEN"
	CMD="python scripts/das_llm.py $DEFAULT_ARGS"

	for REVISION in step0 main; do
		$CMD --model_revision $REVISION --model_name $MODEL --transformation_type "Rotation"
		$CMD --model_revision $REVISION --model_name $MODEL --transformation_type "RevNet"
	done
done



