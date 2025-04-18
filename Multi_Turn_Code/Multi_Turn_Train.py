from itertools import combinations
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    LlamaForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from transformers import Trainer, TrainingArguments as BaseTrainingArguments
from transformers.trainer_utils import get_last_checkpoint

import json
from typing import Any, Dict
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
import functools
from torch.utils.checkpoint import checkpoint
checkpoint_use_reentrant = False

from BinaryLoss import BinaryLoss
from DataLoader import load_datasets, LabelFilter
from Arguments import ScriptArguments, TrainingArguments
from Trainer import PreferenceTrainer
from Metrics import compute_metrics

logger = logging.getLogger(__name__)

class SaMerClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SaMerClassifier, self).__init__()
        self.scoring_layer = nn.Sequential(
            nn.Linear(input_dim, 2048, bias=False), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(2048, 1024, bias=False), nn.SiLU(),
            nn.Linear(1024, 1024, bias=False), nn.SiLU(),
            nn.Linear(1024, 1024, bias=False), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(1024, output_dim, bias=False)
        )

    def forward(self, x):
        mean = self.scoring_layer(x)
        return mean

class DataCollator:
    def __init__(self, args, training_args, tokenizer):
        self.args = args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.pad_token_id = self.tokenizer.pad_token_id
        self.max_length = self.args.max_length

    @torch.no_grad()
    def __call__(self, features: Any) -> Dict[str, Any]:
        # 只处理两个对话
        text_field = ["dialogue_a", "dialogue_b"]
        batch = self.tokenizer(
            sum([[item[text] for text in text_field] for item in features], []),
            add_special_tokens=False, truncation=True, return_tensors="pt", 
            padding=True, max_length=self.max_length
        )
        
        labels = {}
        for label_name in self.args.label_field:
            if label_name in features[0]:
                labels[label_name] = torch.tensor([item[label_name] for item in features], dtype=torch.long)
        
        return dict(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=labels
        )

def confidence_mask(labels, confidence):        # 具体实现置信度过滤的函数，让训练更关注高质量样本
    return (labels - 0.5).abs() >= confidence / 2  


class ConfidenceFilter:         # 置信度过滤 (Confidence Filtering), 只保留预测置信度超过阈值的样本进行训练
    def __init__(self, label_field, confidence):
        self.label_field = label_field
        self.confidence = confidence

    def __call__(self, example):
        labels = torch.tensor([example[label] for label in self.label_field])
        return confidence_mask(labels[labels != -100], self.confidence).any().item()     # 例：confidence=0.8，只有模型预测概率大于0.9或小于0.1的样本才会被保留


def bce_with_temperature(probs, labels, temperature = 2.0):     # 温度缩放 (Temperature scaling), 调节模型预测的"确信度"
    probs = probs.clamp(min=0.0, max=1.0)                       # 温度值越高 (2.0), 预测分布越平滑，模型越"谨慎"; 繁殖模型越"自信"
    labels = labels.clamp(min=0.0, max=1.0)

    if temperature != 1.0:
        labels = (labels.logit() / temperature).sigmoid()

    return torch.nn.functional.binary_cross_entropy(probs, labels)

def main():
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    
    # 确保输出目录存在
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Additional arguments {args}")
    
    # 修改检查点检测和恢复逻辑
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        try:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
                # 验证检查点是否有效
                trainer_state_path = os.path.join(last_checkpoint, "trainer_state.json")
                if os.path.exists(trainer_state_path):
                    logger.info(
                        f"Checkpoint detected, resuming training at {last_checkpoint}."
                    )
                    training_args.resume_from_checkpoint = last_checkpoint
                else:
                    logger.warning(
                        f"Checkpoint directory {last_checkpoint} found but trainer_state.json is missing. "
                        f"Starting training from scratch."
                    )
                    last_checkpoint = None
        except Exception as e:
            logger.warning(f"Error detecting checkpoint: {e}. Starting training from scratch.")
            last_checkpoint = None

    # Set seed before initializing model. 
    # Step 1: 数据准备阶段
    set_seed(training_args.seed)
    print(">>>>>>", args.tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        cache_dir=args.cache_dir,
        use_fast=args.use_fast_tokenizer,
        revision=args.model_revision,
        use_auth_token=True if args.use_auth_token else None,
    )       

    config = AutoConfig.from_pretrained(
        args.config_name,
        cache_dir=args.cache_dir,
        revision=args.model_revision,
        use_auth_token=True if args.use_auth_token else None
    )       # 第一步: 加载tokenizer和配置

    if args.config_overrides:
        logger.info(f"Overriding config: {args.config_overrides}")
        config.update_from_string(args.config_overrides)
        logger.info(f"New config: {config}")

    config.num_labels = 2 * (len(args.label_field) - 1)   # remove "Overall winner"
    tokenizer.pad_token_id = 0
    config.pad_token_id = 0

    # step 2: 模型初始化阶段
    if args.model_name_or_path:
        half_dtype = (torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None))
        device_map = {"":int(os.environ.get("LOCAL_RANK") or 0)}
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            revision=args.model_revision,
            use_auth_token=True if args.use_auth_token else None,
            torch_dtype=half_dtype,
            use_flash_attention_2=training_args.use_flash_attention_2,
        )

        # 2. freeze all parameters （冻结基础模型参数)
        for param in model.model.parameters():
            param.requires_grad = False
    else:
        model = AutoModelForSequenceClassification.from_config(config)

    if args.lora or args.lora_path:
        from peft import PeftModel, get_peft_model, LoraConfig, TaskType
        if args.lora_path:
            logger.info(f">>>>>> Loading LoRA model from {args.lora_path}")
            model = PeftModel.from_pretrained(model, args.lora_path)
        else:
            lora_target_modules = args.lora_target_modules.split(',')
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=lora_target_modules,
                modules_to_save=args.lora_modules_to_save,
            )
            model = get_peft_model(model, peft_config)
        
        model.print_trainable_parameters()

    logger.info(f"Model: {model}")

    train_dataset = {}
    val_dataset = {}
    
    train_paths = Path(args.train_datasets_dir)
    train_files = [str(f) for f in train_paths.glob("**/*.json")]
    
    if not train_files:
        raise ValueError(f"No JSON files found in {args.train_datasets_dir}. Check your directory path.")
    
    print(f"Found {len(train_files)} training files: {train_files}")
    
    train_dataset, val_dataset = load_datasets(
        tokenizer,
        train_files,
        args.eval_split_size_train if args.eval_split_size_train is not None else args.eval_split_size,
        training_args.seed,
        args.label_field,
        training_args.dataloader_num_workers,
        cache_dir=args.cache_dir
    )

    # 检查数据集是否为空
    if not train_dataset or all(len(ds) == 0 for ds in train_dataset.values()):
        raise ValueError("Training dataset is empty after processing. Check your data and filtering criteria.")
    
    train_dataset = datasets.concatenate_datasets(list(train_dataset.values()))
    val_dataset = datasets.concatenate_datasets(list(val_dataset.values()))
    
    print(f"Initial dataset sizes - Train: {len(train_dataset)}, Validation: {len(val_dataset)}")

    logger.warning(f"After confidence filtering - train sequences: {len(train_dataset):,} - validation sequences: {len(val_dataset):,}")

    if training_args.do_eval:
        eval_paths = Path(args.eval_datasets_dir)
        eval_files = [os.path.join(eval_paths, f) for f in eval_paths.glob("**/*.json")]

        _, eval_dataset = load_datasets(
            tokenizer,
            eval_files,
            args.eval_split_size,
            training_args.seed,
            args.label_field,
            training_args.dataloader_num_workers,
            cache_dir=args.cache_dir
        )

        eval_dataset["all"] = datasets.concatenate_datasets(list(eval_dataset.values()))
        logger.warning(f"All eval sequences: {len(eval_dataset['all']):,}")

        eval_dataset["validation"] = val_dataset
    # Step 3: 数据预处理阶段 (训练器设置阶段)
    # 3.1 创建数据整理器
    collator = DataCollator(args, training_args, tokenizer)

    # 3.2 Initialize our Trainer（初始化训练器）
    trainer = PreferenceTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        label_field=args.label_field, 
    )

    if trainer.is_fsdp_enabled:
        def layer_policy_fn(module):
            return "layer" in module.__class__.__name__.lower()

        auto_wrap_policy = functools.partial(lambda_auto_wrap_policy,
                                             lambda_fn=layer_policy_fn)
        trainer.accelerator.state.fsdp_plugin.auto_wrap_policy = auto_wrap_policy

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
            print(f"Checkpoint detected, resuming training at {checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=None)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset)
        metrics["eval_samples"] =len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()

