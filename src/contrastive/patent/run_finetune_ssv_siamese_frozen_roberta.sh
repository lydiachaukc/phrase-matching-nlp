#!/bin/bash
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --export=NONE
BATCH=64
LR=00005
TEMP=007
AUG="all"
PREAUG="all"
python3 run_finetune_siamese.py \
    --train_file="/w/284/lydiachau/phrase-matching-nlp/data/processed/patent/contrastive/patent-train.pkl.gz" \
    --do_train \
	--dataset_name=amazon-google \
    --train_file "/w/284/lydiachau/phrase-matching-nlp/data/interim/patent/patent-train.json.gz" \
	--validation_file  "/w/284/lydiachau/phrase-matching-nlp/data/interim/patent/patent-train.json.gz" \
	--test_file "/w/284/lydiachau/phrase-matching-nlp/data/interim/patent/patent-train.json.gz" \
	--evaluation_strategy=epoch \
	--tokenizer="roberta-base" \
	--grad_checkpoint=True \
    --output_dir /your_path/contrastive-product-matching/reports/contrastive-ft-siamese/patent-ssv-$AUG$BATCH-$PREAUG$LR-$TEMP-frozen-roberta-base/ \
	--per_device_train_batch_size=128 \
	--per_device_eval_batch_size=128 \
	--learning_rate=5e-05 \
	--weight_decay=0.01 \
	--num_train_epochs=50 \
	--lr_scheduler_type="linear" \
	--warmup_ratio=0.05 \
	--max_grad_norm=1.0 \
	--fp16 \
	--metric_for_best_model=f1 \
	--dataloader_num_workers=1 \
	--disable_tqdm=True \
	--save_strategy="epoch" \
	--load_best_model_at_end \
	--augment=$AUG \
	#--do_param_opt \