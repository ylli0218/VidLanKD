# The name of experiment
GPUS=$1
NAME=$2
echo $2
echo $3
# Create dirs and make backup
teacher_checkpoint=/dccstor/sentient1/git/VidLanKD/snap/vlm/bz128X1X4_trainZh_conRatio3/checkpoint-epoch0019-step000194244/

export TRAIN_FILE=/u/yulongl/.conda/envs/voken/vokenization/data/wiki-cased/en.train.raw
export TEST_FILE=/u/yulongl/.conda/envs/voken/vokenization/data/wiki-cased/en.valid.raw

# Pre-training
# for vRatio in 1 3 10 30 100
for vRatio in 30
do
    sleep 10
    output=/dccstor/phalanx/yulong/experiments/${NAME}_vRatio${vRatio}
    mkdir -p $output/src
    cp -r vlm $output/src/
    cp scripts/run_glue_at_epoch.bash $output/run_glue_at_epoch.bash 
    cp $0 $output/run.bash
    jbsub -q x86_24h -cores 1+4 -mem 300g -require 'a100&&hname!=cccxc536&&hname!=cccxc539&&hname!=cccxc534' -proj bert-base -name vRatio${vRatio} \
    " \
    python vlm/run_vlm_distributed.py \
        --output_dir=$output \
        --overwrite_output_dir \
        --config_name=vlm/configs/bert-12L-768H.json \
        --tokenizer_name=bert-base-uncased \
        --model_type=bert \
        --block_size=126 \
        --per_gpu_train_batch_size=128 \
        --per_gpu_eval_batch_size=128 \
        --gradient_accumulation_steps=1 \
        --max_steps=500000 \
        --num_train_epochs=20 \
        --learning_rate=2e-4 \
        --weight_decay=0.01 \
        --warmup_steps=5000 \
        --mlm_probability 0.15 \
        --voken_ratio ${vRatio} \
        --do_kd1_objective \
        --do_train \
        --train_data_file=$TRAIN_FILE \
        --do_eval \
        --eval_data_file=$TEST_FILE \
        --col_data \
        --save_steps=20000 \
        --split_sent \
        --teacher_dir ${teacher_checkpoint} \
        --mlm ${@:3} > $output/log.log 2>&1 "
done

# --do_kd1_objective mmd KD loss
# --do_kd2_objective CRD KD loss
    # --fp16 \
    # --fp16_opt_level O2 \
    # --teacher_dir ${teacher_checkpoint} \
    # --do_kd1_objective \