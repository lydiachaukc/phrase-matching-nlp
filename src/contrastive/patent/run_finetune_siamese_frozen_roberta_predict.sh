#!/bin/bash
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --export=NONE
BATCH=64
LR=0.00001
TEMP=007
AUG="all"
PREAUG="all"
python3 run_finetune_siamese.py \
	--model_pretrained_checkpoint="/w/284/lydiachau/phrase-matching-nlp/reports/contrastive/us-patern-clean-all128-0.00001-0.07-roberta-base/checkpoint-570/pytorch_model.bin" \
    --do_train \
	--do_eval \
	--do_predict \
	--frozen=False \
	--dataset_name=patent \
    --train_file="/w/284/lydiachau/phrase-matching-nlp/data/interim/patent/patent-train.json.gz" \
	--validation_file="/w/284/lydiachau/phrase-matching-nlp/data/interim/patent/patent-train.json.gz" \
	--test_file="/w/284/lydiachau/phrase-matching-nlp/data/interim/patent/patent-train.json.gz" \
	--evaluation_strategy=epoch \
	--tokenizer="roberta-base" \
	--grad_checkpoint=False \
    --output_dir="/w/284/lydiachau/phrase-matching-nlp/reports/contrastive-ft-siamese/patent-clean-$AUG$BATCH-$PREAUG$LR-$TEMP-frozen-roberta-base/" \
	--per_device_train_batch_size=64 \
	--per_device_eval_batch_size=64 \
	--learning_rate=$LR \
	--weight_decay=0.01 \
	--num_train_epochs=20 \
	--lr_scheduler_type="linear" \
	--warmup_ratio=0.05 \
	--max_grad_norm=1.0 \
	--fp16 \
	--metric_for_best_model=loss \
	--dataloader_num_workers=1 \
	--disable_tqdm=True \
	--augment=$AUG \
	--save_strategy="epoch" \
	--logging_strategy "epoch" \
	--overwrite_output_dir=True \
	--load_best_model_at_end

	# --save_strategy="steps" \
	# --save_steps=2000 \