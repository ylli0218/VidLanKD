#!/bin/bash

# post-fix of run

export NUM_MAX_THREADS=160
export NUMEXPR_MAX_THREADS=$NUM_MAX_THREADS
export MASTER_PORT=12578
GPUS=$1
NAME=$2
GPUS_PER_NODE=$(echo $1 | awk -F '[0-9]' '{print NF-1}' )

# Pre-training
# for conRatio in 0.3 1 3 10
for conRatio in 1 10
do
    sleep 5
    # Create dirs and make backup
    output=snap/vlm/${NAME}_conRatio${conRatio}
    mkdir -p $output/src
    cp -r vlm $output/src/
    cp scripts/run_glue_at_epoch.bash $output/run_glue_at_epoch.bash 
    cp $0 $output/run.bash
    # CHECKPOINT=$(ls -d ${PWD}/${output}/checkpoint-epoch* | tail -n 1)
    # CHECKPOINT=/dccstor/sentient1/git/VidLanKD/snap/vlm/bz128X2_cls_conRatio0.1/checkpoint-epoch0000-step000005000
    # echo "find checkpoint: ${CHECKPOINT}"
    log_out="${output}/log/%x_%j_task-%3t.out"
    log_err="${output}/log/%x_%j_task-%3t.err"
    mkdir -p "${output}/log"
    TRAIN_CMD=" \
    CUDA_VISIBLE_DEVICES=$1 python3 vteacher/run_vlm_distributed.py \
        --output_dir=$output \
        --overwrite_output_dir \
        --config_name=vlm/configs/bert_base.json \
        --tokenizer_name=bert-base-uncased \
        --model_type=bert \
        --block_size=126 \
        --per_gpu_train_batch_size=128 \
        --per_gpu_eval_batch_size=128 \
        --gradient_accumulation_steps=2 \
        --num_train_epochs=20 \
        --learning_rate=5e-5 \
        --weight_decay=0.01 \
        --warmup_steps=0 \
        --mlm_probability 0.15 \
        --mlm_ratio 1.0 \
        --contra_ratio ${conRatio} \
        --do_train \
        --do_eval \
        --col_data \
        --split_sent \
        --shuffle \
        --voken_labels all \
        --dim 64 \
        --voken_hinge_loss \
        --secLang_type='hfl/chinese-macbert-base' \
        --save_steps=5000 \
        --use_CLS_token \
        --mlm "
    SLURM_CMD="srun \
      -o "${log_out}" \
      -e "${log_err}" \
      $TRAIN_CMD \
      "
    sbatch \
      -J ${NAME} \
      -o sbatch_%x_%j.out \
      -e sbatch_%x_%j.err \
      --workdir=${output} \
      --gres=gpu:$GPUS_PER_NODE \
      --nodes=1 \
      --ntasks-per-node=1 \
      --cpus-per-task=$NUM_MAX_THREADS \
      --qos=npl-1hr \
      --time=12:00:00 \
      ${SLURM_CMD}
done