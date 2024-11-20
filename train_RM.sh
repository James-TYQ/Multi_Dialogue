#!/bin/bash

# > Default arguments - can be overriden by environment variables:
# architecture to train, must be compatible with the Llama architecture
model=${MODEL:-"RLHFlow/ArmoRM-Llama3-8B-v0.1"}
# total batch size across all devices with gradient accumulation
bsz=${BSZ:-32}
# number of sequences per device
seq=${SEQ:-1}
# peak learning rate
lr=${LR:-5e-5}
# number of epochs
epochs=${EPOCHS:-1}
# warmup ratio
warmup=${WARMUP:-0.1}
# save model every n steps
save_steps=${SAVE:-5000}
# suffix to append to run name
suffix=${SUFFIX:-"rebuttal-0.5-1"}
# only predict labels with certain confidence
confidence=${CONFIDENCE:-0.0}
# temperature applied to labels
labeltemp=${LABELTEMP:-2.0}
# which labels to predict
label_index=${LABELINDEX:-"all"}

num_gpus=${NUM_GPUS:-2}

deepspeed_config_file="config/ds_zero2_qwen2.json"

run_name="qurater_$(basename $model)_bsz${bsz}_lr${lr}_epochs${epochs}_warmup${warmup}_conf${confidence}_labeltemp${labeltemp}${suffix}"
out_dir="./results/$run_name"
mkdir -p $out_dir

header="deepspeed --include localhost:0,1 --master_port 8333 train_preference_model.py \
--deepspeed $deepspeed_config_file"


# export OMP_NUM_THREADS=$num_gpus

export WANDB_PROJECT="lm-data-selection"
export WANDB_DIR=$out_dir
# export WANDB_MODE="offline"
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export DS_SKIP_CUDA_CHECK=1

# export FSDP_SHARDING_STRATEGY="5" # 5 corresponds to _hybrid_shard_zero2
# export FSDP_STATE_DICT_TYPE="FULL_STATE_DICT"


base_arguments=(
    --report_to wandb

    --do_eval
    --do_train
    --model_name_or_path $model
    --config_name $model
    --config_overrides ""
    --tokenizer_name $model

    --run_name $run_name
    --output_dir $out_dir
    --log_level info
    --logging_steps 5
    --disable_tqdm false
    --save_strategy "steps"
    --save_steps $save_steps
    --eval_strategy "steps"
    --eval_steps $(($save_steps / 1))
    --load_best_model_at_end true
    --metric_for_best_mode eval_all_rank_acc
    --greater_is_better true
    --dataloader_num_workers 2
    --cache_dir .cache
    --overwrite_output_dir
    --remove_unused_columns false
    --use_fast_tokenizer false
    --gradient_checkpointing true

    --num_train_epochs $epochs
    --max_length 8192
    --per_device_train_batch_size $seq
    --per_device_eval_batch_size 1
    --gradient_accumulation_steps $(($bsz / $seq / $num_gpus))
    --learning_rate $lr
    --max_grad_norm 1.0
    --weight_decay 0.1
    --warmup_ratio $warmup

    --use_flash_attention_2 true
    --bf16_full_eval
    --bf16
    --ddp_find_unused_parameters false
    --ddp_timeout 36000000
    # --fsdp auto_wrap  # DO NOT USE WITH deepspeed

    # Depending on model size and sequence length, gradient checkpointing might result in higher throughput
    # --gradient_checkpointing

    # --label_field $label_field
    --confidence_threshold $confidence
    --label_temperature $labeltemp

    --train_datasets_dir data/train
    --eval_split_size_train 0.01

    --eval_datasets data/eval
    --eval_split_size 1.0

    $@
)

echo command: "${header} ${base_arguments[@]}"
${header} "${base_arguments[@]}" # 2>&1 | tee -a $out_dir/log.out