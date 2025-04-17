#!/bin/bash

# > Default arguments - can be overriden by environment variables:
# architecture to train, must be compatible with the Llama architecture
model=${MODEL:-"RLHFlow/ArmoRM-Llama3-8B-v0.1"}
# total batch size across all devices with gradient accumulation
bsz=${BSZ:-16}
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
suffix=${SUFFIX:-"multi-turn-dialogue"}
# only predict labels with certain confidence
confidence=${CONFIDENCE:-0.0}
# temperature applied to labels
labeltemp=${LABELTEMP:-2.0}
# 敏感度和特异度学习率
ss_lr=${SS_LR:-5e-6}
# 初始敏感度和特异度值
initial_sensitivity=${INIT_SENS:-0.8}
initial_specificity=${INIT_SPEC:-0.6}

num_gpus=${NUM_GPUS:-1}

# 注释掉DeepSpeed配置文件
# deepspeed_config_file="config/ds_zero2_qwen2.json"

run_name="multi_turn_$(basename $model)_bsz${bsz}_lr${lr}_ss_lr${ss_lr}_sens${initial_sensitivity}_spec${initial_specificity}_epochs${epochs}_warmup${warmup}_conf${confidence}_labeltemp${labeltemp}${suffix}"
out_dir="./results/$run_name"
mkdir -p $out_dir

# header="deepspeed --include localhost:0 --master_port 8333 Multi_Turn_Train.py \
# --deepspeed $deepspeed_config_file"
header="python  Multi_Turn_Train.py"

export WANDB_PROJECT="multi-turn-dialogue-evaluation"
export WANDB_DIR=$out_dir
# export WANDB_MODE="offline"

base_arguments=(
    --report_to wandb

    --do_eval
    --do_train
    --model_name_or_path $model
    
    --initial_sensitivity $initial_sensitivity
    --initial_specificity $initial_specificity
    --ss_learning_rate $ss_lr
    
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
    --metric_for_best_mode eval_preference_acc
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

    --label_field "结论"
    
    --confidence_threshold $confidence
    --label_temperature $labeltemp

    --train_datasets_dir /mnt/e/SaMer/SaMer/Multi_Turn_Train_Data/train
    --eval_split_size_train 0.01

    --eval_datasets /mnt/e/SaMer/SaMer/Multi_Turn_Train_Data/eval
    --eval_split_size 1.0

    $@
)

echo command: "${header} ${base_arguments[@]}"
${header} "${base_arguments[@]}" # 2>&1 | tee -a $out_dir/log.out