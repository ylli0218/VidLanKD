# The name of experiment
GPUS=$1

NAME=$2

# Create dirs and make backup
output=snap/vlm/$NAME
mkdir -p $output/src
cp -r vlm $output/src/
cp scripts/run_glue_at_epoch.bash $output/run_glue_at_epoch.bash 
cp $0 $output/run.bash

# Pre-training
jbsub -q x86_24h -cores 1+2 -mem 80g -require a100 -proj parallel_data -name train_pc \
" \
CUDA_VISIBLE_DEVICES=$1 python3 vteacher/run_vlm_distributed.py \
    --output_dir=$output \
	--overwrite_output_dir \
	--config_name=vlm/configs/bert-6L-768H.json \
	--tokenizer_name=bert-base-uncased \
    --model_type=bert \
	--block_size=126 \
	--per_gpu_train_batch_size=32 \
    --per_gpu_eval_batch_size=32 \
	--gradient_accumulation_steps=4 \
	--num_train_epochs=20 \
	--learning_rate=5e-5 \
	--weight_decay=0.01 \
	--warmup_steps=0 \
    --mlm_probability 0.15 \
    --mlm_ratio 1.0 \
    --do_train \
    --do_eval \
    --col_data \
    --split_sent \
    --voken_labels all \
    --dim 64 \
    --voken_hinge_loss \
    --secLang_type='hfl/chinese-macbert-base' \
    --mlm ${@:3} > $output/log.log 2>&1 "


    # --secLang_type='hfl/chinese-macbert-base' \
# CUDA_VISIBLE_DEVICES=$1 python3 -m debugpy --listen "${HOSTNAME}:15678" --wait-for-client vteacher/run_vlm_distributed.py \
