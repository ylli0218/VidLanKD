export GLUE_DIR=data/glue/
EPOCHS=$1
MODEL=$2
CKPT=/dccstor/sentient1/git/llm-eval-with-raytune/xlm-align-base
for TASK_NAME in CoLA 
do
    for SEED in {38..47}
    do
        OUTDIR=/dccstor/sentient1/git/VidLanKD/snap/glue/$MODEL/$TASK_NAME/seed${SEED}
        mkdir -p ${OUTDIR}
        cp $0 >${OUTDIR}/run.bash
        jbsub -q x86_1h -cores 1+1 -mem 80g -require a100 -proj glue -name run_glue \
        " \
        python vlm/run_glue.py \
            --model_type xlmroberta \
            --model_name_or_path $CKPT \
            --task_name $TASK_NAME \
            --do_train \
            --do_eval \
            --do_lower_case \
            --data_dir $GLUE_DIR/$TASK_NAME \
            --save_steps -1 \
            --max_seq_length 126 \
            --per_gpu_eval_batch_size=128   \
            --per_gpu_train_batch_size=128   \
            --learning_rate 2e-5 \
            --warmup_steps 0.1 \
            --num_train_epochs $EPOCHS.0 \
            --overwrite_output_dir \
            --output_dir ${OUTDIR} \
            > ${OUTDIR}/log.log 2>&1 "
    done
done
            #--overwrite_output_dir \
            # --tokenizer_name=bert-base-uncased \
