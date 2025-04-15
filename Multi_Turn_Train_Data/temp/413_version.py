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

logger = logging.getLogger(__name__)


class SaMerClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SaMerClassifier, self).__init__()
        # 得分预测层
        self.scoring_layer = nn.Sequential(
            nn.Linear(input_dim, 2048, bias=False), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(2048, 1024, bias=False), nn.SiLU(),
            nn.Linear(1024, 1024, bias=False), nn.SiLU(),
            nn.Linear(1024, 1024, bias=False), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(1024, output_dim, bias=False)
        )
        # 方差预测层 
        self.variance_layer = nn.Sequential(
            nn.Linear(input_dim, 1024, bias=False), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(1024, 1024, bias=False), nn.SiLU(),
            nn.Linear(1024, 1024, bias=False), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(1024, output_dim, bias=False)
        )

    def forward(self, x):
        mean = self.scoring_layer(x)
        var = F.softplus(self.variance_layer(x)) + 1e-6  # 使用softplus确保方差为正值
        return mean, var

# class LlamaForMDQRwardModel(AutoModelForSequenceClassification):
#     def __init__(self, config):
#         super().__init__(config)
#         self.score = SaMerClassifier(config.hidden_size, config.num_labels//2)

#     def forward(self, input_ids=None, attention_mask=None, **kwargs):
#         outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
#         last_hidden_state = outputs.last_hidden_state
        
#         # 获取每个序列的最后一个非填充token的隐藏状态
#         sequence_lengths = attention_mask.sum(dim=1) - 1
#         batch_size = last_hidden_state.shape[0]
#         batch_indices = torch.arange(batch_size, device=last_hidden_state.device)
#         last_token_hidden_states = last_hidden_state[batch_indices, sequence_lengths]
        
#         # 获取预测结果并分割为均值和方差
#         predictions = self.score(last_token_hidden_states)
#         output_dim = predictions.shape[1] // 2
#         mean = predictions[:, :output_dim]  # 前半部分为均值
#         var_raw = predictions[:, output_dim:]  # 后半部分为方差
        
#         # 使用softplus确保方差为正值
#         var = F.softplus(var_raw) + 1e-6
        
#         print("预测分数:", mean.shape, mean.mean().item())
#         print("预测方差:", var.shape, var.mean().item())
        
#         # 直接返回均值和方差作为元组
#         return mean, var

@dataclass
class ScriptArguments:
    text_field: List[str] = field(
        default_factory=lambda: ["instruction", "dialogue_a", "dialogue_b"],
        metadata={
            "help": "Name of the text field in the dataset."
        },
    )       
    label_field: List[str] = field(
        default_factory=lambda: ['准确性', '逻辑性', '口语性', '关联性', '个性化', '创造性', '互动性', '情感性', '知识性', '安全性', '结论'],
        # default_factory=lambda: ['Accuracy', 'Logic', 'Spokenness', 'Relevance', 'Personalization', 'Creativity', 'Interaction', 'Emotionality', 'Knowledge', 'Security', 'Conclusion'],
        metadata={
            "help": "Name of the label field in the dataset."
        },
    )
    train_datasets_dir: str = field(
        default=None,     
        metadata={"help": "Directory of training datasets."},
    )
    eval_datasets_dir: str = field(
        default=None,
        metadata={"help": "Path to eval dataset."},
    )
    eval_split_size: float = field(
        default=0.0,
        metadata={"help": "Validation split size."},
    )
    eval_split_size_train: Optional[float] = field(
        default=0.8,
        metadata={"help": "Validation split size for training datasets."},
    )

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    max_length: Optional[int] = field(
        default=2048,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )

    lora: bool = field(default=False, metadata={"help": "Whether to use parameter efficient fine-tuning."})
    lora_path: str = field(default=None, metadata={"help": "Path to the lora model."})
    lora_modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    lora_r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    lora_target_modules: str = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=16, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "Lora dropout"})

    single_label_ablation: int = field(default=-1, metadata={"help": "Whether to use single label for ablation."})

@dataclass
class TrainingArguments(BaseTrainingArguments):
    label_temperature: float = field(
        default=2.0,
        metadata={"help": "Label temperature"},
    )
    log_confidences: List[float] = field(
        default_factory=lambda: [0.5, 0.8],
        metadata={"help": "Confidence thresholds for logging accuracy"},
    )
    confidence_threshold: float = field(
        default=0.0,
        metadata={
            "help": "Confidence threshold for including data during training"
        },
    )
    use_flash_attention_2: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use flash attention 2"
        },
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use gradient checkpointing"
        },
    )
    output_dir: str = field(
        default='',
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    dataloader_drop_last: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to drop the last incomplete batch."
        },
    )
    # 添加敏感度和特异度的初始值参数
    initial_sensitivity: float = field(
        default=0.5,
        metadata={"help": "Initial value for sensitivity parameter (0-1)"},
    )
    initial_specificity: float = field(
        default=0.5,
        metadata={"help": "Initial value for specificity parameter (0-1)"},
    )
    ss_learning_rate: float = field(
        default=5e-6,
        metadata={"help": "Learning rate for sensitivity and specificity parameters"},
    )

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
        
        # 创建标签张量
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

class LabelFilter:
    def __init__(self, label_field):
        self.label_field = label_field

    def __call__(self, example):
        # 降低标准：只要有一个标签不是-100就保留
        for label in self.label_field:
            if label in example and isinstance(example[label], torch.Tensor):
                # 确保是张量且非空
                if len(example[label]) > 0 and not (example[label] == -100).all().item():
                    return True
        # 如果至少有对话数据，也保留
        return ('dialogue_a' in example and 'dialogue_b' in example and 
                example['dialogue_a'] and example['dialogue_b'])


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

class BinaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, probs, labels, sensitivity, specificity):
        
        # 处理维度不匹配的情况
        if len(probs.shape) == 1:
            probs = probs.unsqueeze(1)  # 变成 [batch_size, 1]
        
        if len(labels.shape) == 1:
            labels = labels.unsqueeze(1)  # 变成 [batch_size, 1]
        
        # 创建掩码，排除平局样本(标签为2)和无效标签
        valid_mask = (labels == 0) | (labels == 1)
        
        if valid_mask.sum() == 0:
            # 如果没有有效样本，返回零损失
            return torch.tensor(0.0, device=probs.device), torch.tensor(0.0, device=probs.device)
        
        # 处理多个标注者的情况
        if probs.shape[1] == 1 and labels.shape[1] > 1:
            probs = probs.expand(-1, labels.shape[1])
        
        # 只保留有效标签样本
        valid_probs = probs[valid_mask]
        valid_labels = labels[valid_mask].float()
        
        # 确保标签值在[0,1]范围内
        valid_labels = torch.clamp(valid_labels, 0, 1)
        
        # 使用Binary_Loss的方法计算损失
        # 将标签转换为one-hot编码
        g = torch.zeros((valid_labels.size(0), 2), device=valid_labels.device)
        g.scatter_(1, valid_labels.long().unsqueeze(1), 1)
        
        # 将预测概率转换为二分类格式 [p, 1-p]
        p_binary = torch.stack([1-valid_probs, valid_probs], dim=1)
        
        # 计算敏感性和特异性参数
        alpha = torch.tensor([1-sensitivity, sensitivity], device=probs.device)  # [neg, pos]
        beta = torch.tensor([specificity, 1-specificity], device=probs.device)   # [neg, pos]
        
        # 计算对数似然
        log_a = g * torch.log(alpha) + (1 - g) * torch.log(1 - alpha)
        log_a = torch.sum(log_a, dim=1, keepdim=True)
        a = torch.exp(log_a)  # 正样本对数似然
        
        log_b = (1 - g) * torch.log(beta) + g * torch.log(1 - beta)
        log_b = torch.sum(log_b, dim=1, keepdim=True)
        b = torch.exp(log_b)  # 负样本对数似然
        
        # 计算最终损失
        p_sum = torch.sum(p_binary * g, dim=1, keepdim=True)  # 提取对应标签的预测概率
        loss_val = torch.log(a * p_sum + b * (1 - p_sum)) / 2  # 除以2是因为g的维度为2
        loss = -torch.mean(loss_val)
        
        # 计算准确率
        preds = (valid_probs > 0.5).float()
        accuracy = (preds == valid_labels).float().mean()
        
        return loss, accuracy
    
class PreferenceTrainer(Trainer):
    def __init__(self, *args, label_field=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.binary_loss = BinaryLoss()
        
        # 保存label_field属性
        self.label_field = label_field
        
        # 创建可训练的敏感度和特异度参数，使用args中的初始值
        # 将0-1范围的值转换为logit值
        initial_sensitivity_logit = torch.logit(torch.tensor(self.args.initial_sensitivity))
        initial_specificity_logit = torch.logit(torch.tensor(self.args.initial_specificity))
        
        self.sensitivity = nn.Parameter(initial_sensitivity_logit)
        self.specificity = nn.Parameter(initial_specificity_logit)
        
        # 优化器包含模型参数和敏感度/特异度参数
        optimizer_grouped_parameters = [
            {'params': self.model.parameters()},
            {'params': self.sensitivity, 'lr': self.args.ss_learning_rate},
            {'params': self.specificity, 'lr': self.args.ss_learning_rate}
        ]
        
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate
        )
        
        logger.info(f"初始敏感度: {self.args.initial_sensitivity}, 初始特异度: {self.args.initial_specificity}")

    def compute_loss(self, model, inputs, return_outputs=False, return_output_and_metrics=False):
        # 获取标签和模型输出
        labels = inputs.pop("labels")
        outputs = model(**inputs, use_cache=False)
        print("output", outputs)
        
        # 处理不同类型的输出
        if isinstance(outputs, tuple):
            # 如果outputs是元组，假设第一个元素是logits，第二个是var
            logits = outputs[0]
            var = outputs[1] if len(outputs) > 1 else torch.ones_like(logits) * 1e-6
            print("从元组中获取 - 方差值:", var)
        else:
            # 如果outputs是对象，获取logits
            logits = outputs.logits
            print("outputs类型:", type(outputs), "outputs属性:", dir(outputs))
            # 获取方差值
            if hasattr(outputs, 'var'):
                var = outputs.var
                print("从对象属性中获取 - 方差值:", var)
            else:
                var = torch.ones_like(logits) * 1e-6
                print("使用默认方差值:", var)
        
        print("logits:", logits)
        # 打印A和B回答的方差
        if var.shape[0] >= 2:
            print("A回答方差:", var[0::2])
            print("B回答方差:", var[1::2])
        
        # 获取设备和数据类型
        device = logits.device
        
        # 质量评估损失 (loss_q)
        score_diff = logits[0::2] - logits[1::2]  
        # 初始化总损失
        total_loss = torch.tensor(0.0, device=device)
        
        # 只处理"结论"维度的标签
        conclusion_dim = "结论"
        if conclusion_dim in labels:
            dim_labels = labels[conclusion_dim]
            
            # 使用误差函数计算概率
            score_var = var[0::2]**2 + var[1::2]**2 + 1e-4
            p = 0.5 * (1 + torch.erf(-score_diff / torch.sqrt(2 * score_var)))  # p<0.5, A更好; p>0.5, B更好
            
            print(f"p shape: {p.shape}, dim_labels shape: {dim_labels.shape}")
            # 确保p的形状与dim_labels兼容
            if len(p.shape) == 1 and len(dim_labels.shape) > 1:
                p = p.unsqueeze(1)  # 变成 [batch_size, 1]
            # 使用BinaryLoss计算损失，传入可训练的敏感度和特异度参数
            sensitivity_value = torch.sigmoid(self.sensitivity).item()
            specificity_value = torch.sigmoid(self.specificity).item()
            
            loss, accuracy = self.binary_loss(p, dim_labels, sensitivity=sensitivity_value, specificity=specificity_value)
            total_loss = loss
            
            # 记录损失和准确率
            self.log({
                "loss": total_loss.item(),
                "accuracy": accuracy.item(),
                "sensitivity": sensitivity_value,
                "specificity": specificity_value
            })
        else:
            self.log({"loss": 0.0, "accuracy": 0.0})        # 如果没有结论维度，返回零损失和零准确率
        
        if return_outputs:
            return total_loss, outputs
        return total_loss
        
    def save_model(self, output_dir=None, _internal_call=False):
        # 如果output_dir为None，使用训练参数中的输出目录
        if output_dir is None:
            output_dir = self.args.output_dir
        
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 修改保存参数的方式
        torch.save(
            {
                "sensitivity": self.sensitivity,
                "specificity": self.specificity,
                "sensitivity_value": torch.sigmoid(self.sensitivity).item(),
                "specificity_value": torch.sigmoid(self.specificity).item()
            },
            os.path.join(output_dir, "sensitivity_specificity.pt")
        )
        super().save_model(output_dir, _internal_call)
    
    def _load_from_checkpoint(self, resume_from_checkpoint):
        # 加载模型和敏感度/特异度参数
        super()._load_from_checkpoint(resume_from_checkpoint)
        
        # 尝试加载敏感度和特异度参数
        ss_path = os.path.join(resume_from_checkpoint, "sensitivity_specificity.pt")
        if os.path.exists(ss_path):
            ss_dict = torch.load(ss_path)
            self.sensitivity = ss_dict["sensitivity"]
            self.specificity = ss_dict["specificity"]
            logger.info(f"Loaded sensitivity: {torch.sigmoid(self.sensitivity).item()}, specificity: {torch.sigmoid(self.specificity).item()}")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = torch.tensor(logits)
    
    # 创建维度指标字典
    metrics = {}
    
    # 处理每个维度的标签
    for dim_name, dim_labels in labels.items():
        # 计算评分差异
        score_diff = logits[0::2] - logits[1::2]
        
        # 处理多标注者情况 (dim_labels形状为[batch_size, num_annotators])
        if len(dim_labels.shape) > 1 and dim_labels.shape[1] > 1:
            # 创建新的处理后的标签
            processed_labels = torch.full((dim_labels.shape[0],), -100, dtype=torch.long)
            
            for i in range(dim_labels.shape[0]):
                # 获取当前样本的所有标注
                sample_labels = dim_labels[i]
                # 过滤掉平局情况
                valid_labels = sample_labels[sample_labels != -100]
                
                if len(valid_labels) > 0:
                    # 计算A胜(0)和B胜(1)的数量
                    a_wins = (valid_labels == 0).sum().item()
                    b_wins = (valid_labels == 1).sum().item()

                    if a_wins >= 2:
                        processed_labels[i] = 0  # A胜
                    elif b_wins >= 2:
                        processed_labels[i] = 1  # B胜
            
            # 使用处理后的标签
            dim_labels = processed_labels
        
        # 计算准确率 - 排除无效和平局样本
        valid_mask = (dim_labels != -100) & (dim_labels != 2)
        if valid_mask.sum() > 0:
            correct_predictions = torch.where(
                valid_mask,
                ((score_diff >= 0.0) == (dim_labels <= 0.5)).float(),
                torch.tensor(float('nan'))
            )
            
            accuracy = correct_predictions.nanmean()
            metrics[f"{dim_name}_acc"] = accuracy.item()
    
    # 计算总体准确率（如果有多个维度）
    if len(metrics) > 0:
        metrics["preference_acc"] = sum(metrics.values()) / len(metrics)
    else:
        metrics["preference_acc"] = 0.0
    
    return metrics

'''
函数作用: 为后续模型训练提供规范化的数据输入。
'''
def load_datasets(tokenizer, dataset_paths, eval_split_size, seed, label_field, num_workers, cache_dir):

    def preprocess_function(examples):
        print(label_field)
    # 为每个样本创建结果字典
        results = {key: [] for key in label_field + ['dialogue_a', 'dialogue_b']}
        
        # 处理对话模板
        for conversation, evaluations in zip(examples['conversation'], examples.get('evaluations', [])):
            try:
                # 构建完整多轮对话
                dialogue_a_messages = []
                dialogue_b_messages = []
                
                # 提取对话内容
                for i in range(0, len(conversation)-1, 2):
                    if i+1 < len(conversation):
                        # 用户消息
                        if 'text' in conversation[i]:
                            human_query = conversation[i]['text']
                            dialogue_a_messages.append({'role': 'user', 'content': human_query})
                            dialogue_b_messages.append({'role': 'user', 'content': human_query})
                        
                        # 助手消息
                        if 'A' in conversation[i+1] and 'B' in conversation[i+1]:
                            dialogue_a_messages.append({'role': 'assistant', 'content': conversation[i+1]['A']})
                            dialogue_b_messages.append({'role': 'assistant', 'content': conversation[i+1]['B']})
                
                # 应用聊天模板
                results['dialogue_a'].append(tokenizer.apply_chat_template(dialogue_a_messages, tokenize=False))
                results['dialogue_b'].append(tokenizer.apply_chat_template(dialogue_b_messages, tokenize=False))
                
                # 处理评估结果
                # 初始化每个维度的标签列表
                dimension_labels = {dim: [] for dim in label_field}
                
                # 处理每个标注者的评价
                for eval_data in evaluations:
                    if 'evaluation' in eval_data:
                        try:
                            # 解析评估JSON
                            evaluation = json.loads(eval_data['evaluation'])
                            
                            # 收集每个维度的评价
                            for dim in label_field:
                                if dim in evaluation:
                                    if evaluation[dim] == "A":
                                        dimension_labels[dim].append(0)  # A胜
                                    elif evaluation[dim] == "B":
                                        dimension_labels[dim].append(1)  # B胜
                                    else:  # Fair
                                        dimension_labels[dim].append(-100)  # 忽略平局
                        except Exception as e:
                            print(f"Error parsing evaluation: {e}")
                            continue
                
                # 将每个维度的标签添加到结果中
                for dim in label_field:
                    if dimension_labels[dim]:
                        # 将标签转换为张量
                        labels = torch.tensor(dimension_labels[dim], dtype=torch.long)
                        results[dim].append(labels)
                    else:
                        # 如果没有标签，添加一个空张量
                        results[dim].append(torch.tensor([], dtype=torch.long))
            except Exception as e:
                print(f"Error processing conversation: {e}")
                # 添加空值以保持结果字典的长度一致
                for key in results:
                    if key == 'dialogue_a' or key == 'dialogue_b':
                        results[key].append("")
                    else:
                        results[key].append(torch.tensor([], dtype=torch.long))
        
        return results

    train_datasets = {}
    eval_datasets = {}
    loaded_datasets = {}
    
    # 加载数据集
    for path in dataset_paths:
        try:
            # 修改这里：直接使用path作为数据文件路径
            dataset = datasets.load_dataset("json", data_files=path, cache_dir=cache_dir)
            print(f"====loaded {path} {dataset}")
        except Exception as e:
            print(f"Error loading dataset from {path}: {e}")
            continue
        
        # 应用预处理
        for split, data in dataset.items():
            try:
                dataset[split] = data.map(
                    preprocess_function,
                    batched=True,
                    num_proc=num_workers,
                    remove_columns=data.column_names,  # 动态获取列名
                    keep_in_memory=True,
                    desc="preprocessing new columns on dataset",
                )
            except Exception as e:
                print(f"Error preprocessing dataset {path}/{split}: {e}")
                continue
        
        if isinstance(dataset, datasets.DatasetDict):
            if "train" in dataset and len(dataset) == 1:
                loaded_datasets[path] = dataset["train"]
            else:
                for split, ds in dataset.items():
                    loaded_datasets[f"{path}/{split}"] = ds
        else:
            loaded_datasets[path] = dataset

    # 过滤和划分数据集
    for path, dataset in loaded_datasets.items():
        try:
            try:
                filtered_dataset = dataset.filter(
                    LabelFilter(label_field), 
                    num_proc=num_workers, 
                    keep_in_memory=True
                )              
                print(f"过滤数据集完成。原始数据集: {len(dataset)} 条，过滤后: {len(filtered_dataset)} 条")
            except Exception as e:
                print(f"过滤数据集 {path} 时出错: {e}。使用原始数据集。")
                filtered_dataset = dataset
            
            # 划分训练集和验证集
            if eval_split_size < 1.0:
                splits = filtered_dataset.train_test_split(test_size=eval_split_size, seed=seed)
                train_dataset, eval_dataset = splits["train"], splits["test"]
            else:
                eval_dataset = filtered_dataset
                train_dataset = filtered_dataset.select(range(min(100, len(filtered_dataset))))
            
            dataset_name = os.path.basename(path)
            
            if dataset_name in train_datasets:
                train_datasets[dataset_name] = datasets.concatenate_datasets([train_datasets[dataset_name], train_dataset])
                eval_datasets[dataset_name] = datasets.concatenate_datasets([eval_datasets[dataset_name], eval_dataset])
            else:
                train_datasets[dataset_name] = train_dataset
                eval_datasets[dataset_name] = eval_dataset
        except Exception as e:
            print(f"处理数据集 {path} 时出错: {e}")
            continue
    
    return train_datasets, eval_datasets


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

