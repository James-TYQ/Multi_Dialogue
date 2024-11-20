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

f = open('./outputs/scenario_weights/logistic_regression.json', 'r')
WEIGHT_DICT = json.load(f)


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
        self.dimpred_layer = nn.Sequential(
            nn.Linear(input_dim, 1024, bias=False), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(1024, 1024, bias=False), nn.SiLU(),
            nn.Linear(1024, 1024, bias=False), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(1024, output_dim, bias=False)
        )
        self.weighting_layer = nn.Sequential(
            nn.Linear(input_dim, 1024, bias=False), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(1024, 1024, bias=False), nn.SiLU(),
            nn.Linear(1024, 1024, bias=False), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(1024, output_dim, bias=False)
        )

    def forward(self, x):
        scores = self.scoring_layer(x)
        dim_prob = self.dimpred_layer(x)
        weights = self.weighting_layer(x)
        return torch.cat([scores, dim_prob, weights], dim=-1)

class LlamaForMDQRwardModel(AutoModelForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.score = SaMerClassifier(config.hidden_size, config.num_labels//2)

@dataclass
class ScriptArguments:
    text_field: List[str] = field(
        default_factory=lambda: ["instruction", "dialogue_a", "dialogue_b"],
        metadata={
            "help": "Name of the text field in the dataset."
        },
    )
    label_field: List[str] = field(
        default_factory=lambda: ['Accuracy', 'Admit Uncertainty', 'Audience Friendly', 'Authenticity', 'Citation', 'Clarity', 'Coverage', 'Creativity', 'Depth', 'Feasibility', 'Harmlessness', 'Information Richness', 'Insight', 'Logic', 'Multiple Aspects', 'Objectivity', 'Originality', 'Professionalism', 'Relevance', 'Timeliness', 'Attractive', 'Interactivity', 'Professional', 'Being Friendly', 'Coherence', 'Emojis', 'Emotion', 'Length', 'Style', 'Vivid', 'Step by Step Explanation', 'Code Correctness', 'Code Readability', 'Instruction Following', 'Layout', 'Modularity', 'Pacing', 'Completeness', 'Faithfulness', 'Pointing Out', 'Result at the Beginning', 'Attractiveness', 'Overall winner'],
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

        # step 1: tokenize the question and the two QA pairs.
        # text_field: ['instruction', 'dialog a', 'dialog b']
        batch = self.tokenizer(
            sum([[item[text] for text in self.args.text_field] for item in features], []),
            add_special_tokens=False, truncation=True, return_tensors="pt", padding=True, max_length=self.max_length)
        bsz = batch.input_ids.size(0)

        num_labels = len(self.args.label_field)
        rk_labels = -100 * torch.ones(bsz//len(self.args.text_field), num_labels, dtype=torch.float32)
        w_labels = -100 * torch.ones(bsz, num_labels-1, dtype=torch.float32)    # ignore "Overall winner"
        
        # step 2: label the dimension-level winner, i.e., rk_labels; and the selected dimension, i.e., w_labels
        counter = 0
        for id, item in enumerate(features):
            k = len(self.args.text_field)
            # step 2.1: label the selected dimension
            w_label = [0 if item[label][0][1] == -100 else 1 for label in self.args.label_field[:-1]]

            # step 2.2: label the dimension-level winner
            for i, label in enumerate(self.args.label_field):
                if item[label][0][1] != -100:
                    rk_labels[id, i] = float(item[label][0][1]) if item[label][0][1] > 0 else -1.0
                    assert rk_labels[id, i] in [-1.0, 1.0, 2.0], f"rk_labels mismatch."

            w_labels[counter:counter + k, :] = torch.tensor(w_label, dtype=torch.float32).unsqueeze(0).repeat(k, 1)
            
            counter += k
        assert (w_labels == -100).float().sum() == 0, f"w_labels error, {(w_labels == -100).float().sum()}"

        return dict(input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    labels={'rk_labels': rk_labels, 'w_labels': w_labels})


def confidence_mask(labels, confidence):
    return (labels - 0.5).abs() >= confidence / 2

class LabelFilter:
    def __init__(self, label_field):
        self.label_field = label_field

    def __call__(self, example):
        labels = torch.tensor([example[label] for label in self.label_field])
        return not (labels == -100).all().item()


class ConfidenceFilter:
    def __init__(self, label_field, confidence):
        self.label_field = label_field
        self.confidence = confidence

    def __call__(self, example):
        labels = torch.tensor([example[label] for label in self.label_field])
        return confidence_mask(labels[labels != -100], self.confidence).any().item()


def bce_with_temperature(probs, labels, temperature = 2.0):
    probs = probs.clamp(min=0.0, max=1.0)
    labels = labels.clamp(min=0.0, max=1.0)

    if temperature != 1.0:
        labels = (labels.logit() / temperature).sigmoid()

    return torch.nn.functional.binary_cross_entropy(probs, labels)

import torch
import torch.nn.functional as F

def multilabel_categorical_crossentropy(preds, labels):
    """
    multilabel categorical crossentropy loss for weight prediction

    Args:
        preds: the prediction of the model. DO NOT use sigmoid and softmax here
        labels: the ground truth label

    Returns:
        loss: the crossentropy loss
    """
    preds = (1 - 2 * labels) * preds
    preds_neg = preds - labels * 1e12
    preds_pos = preds - (1 - labels) * 1e12
    zeros = torch.zeros_like(preds[..., :1])
    preds_neg = torch.cat([preds_neg, zeros], dim=-1)
    preds_pos = torch.cat([preds_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(preds_neg, dim=-1)
    pos_loss = torch.logsumexp(preds_pos, dim=-1)
    return (neg_loss + pos_loss).mean()


class PreferenceTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, return_output_and_metrics=False):

        # rk_labels: the dimension-level winner; 
        # w_labels: the selected dimension
        labels = inputs.pop("labels")
        rk_labels = labels.pop("rk_labels")
        w_labels = labels.pop("w_labels")

        # step 1: get the scores, dim_prob, and weights
        outputs = model(**inputs, use_cache=False)  # outputs.size = (3 * bsz, num_labels x 3)
        scores, dim_prob, weights = torch.split(outputs.logits, outputs.logits.size(-1)//3, dim=-1)

        # step 2: caclucate the dimension prediction loss
        valid_mask = (w_labels != -100)
        loss_dim_pred = bce_with_temperature(dim_prob[valid_mask].float().sigmoid(), w_labels[valid_mask].float(), self.args.label_temperature)

        # the dim_prob is only related to the instruction, not to the dialogue 1 and dialogue 2
        dim_prob_clone = dim_prob.clone()
        dim_prob_clone[1::3] = dim_prob[0::3]
        dim_prob_clone[2::3] = dim_prob[0::3]
        dim_prob = dim_prob_clone

        # step 3. calculate the rank loss, and mse loss for tie
        valid_mask = (rk_labels[:, :-1] != -100) & confidence_mask(rk_labels[:, :-1], self.args.confidence_threshold) & (rk_labels[:, -1:] < 2.0)

        rk_weight_1 = valid_mask.sum().item()
        loss_rk = F.margin_ranking_loss(scores[1::3][valid_mask], scores[2::3][valid_mask], rk_labels[:, :-1][valid_mask], margin=0.3, reduction="mean")
        
        valid_mask = (rk_labels[:, :-1] != -100) & (rk_labels[:, :-1] >= 2.0)
        rk_weight_2 = valid_mask.sum().item()
        if scores[1::3][valid_mask].numel() > 0:
            rk_weight_total = rk_weight_1 + rk_weight_2
            rk_weight_1 = rk_weight_1 / rk_weight_total
            rk_weight_2 = rk_weight_2 / rk_weight_total
            loss_rk = rk_weight_1 * loss_rk + rk_weight_2 * F.mse_loss(scores[1::3][valid_mask], scores[2::3][valid_mask], reduction="mean")

        # step 4. calculate the overall winner rank loss
        ## 4.1 scores range to [0,1]
        scores = scores.sigmoid()
        # 4.2 dimensional softmax
        masked_weights = torch.where(w_labels.bool(), weights, torch.tensor(float('-inf'), device=weights.device))
        masked_weights = masked_weights.softmax(dim=-1)

        overall_score = (masked_weights * scores).sum(dim=-1, keepdim=True)


        valid_mask = (rk_labels[:, -1:] != -100) & (rk_labels[:, -1:] < 2.0)
        overall_w_1 = valid_mask.sum().item()
        loss_overall = F.margin_ranking_loss(overall_score[1::3][valid_mask], overall_score[2::3][valid_mask], rk_labels[:, -1:][valid_mask], margin=0.3, reduction="mean")

        valid_mask = (rk_labels[:, -1:] != -100) & (rk_labels[:, -1:] >= 2.0)
        overall_w_2 = valid_mask.sum().item()
        if overall_score[1::3][valid_mask].numel() > 0:
            overall_w_total = overall_w_1 + overall_w_2
            overall_w_1 = overall_w_1 / overall_w_total
            overall_w_2 = overall_w_2 / overall_w_total
            loss_overall = overall_w_1 * loss_overall + overall_w_2 * F.mse_loss(overall_score[1::3][valid_mask], overall_score[2::3][valid_mask], reduction="mean")

        loss = loss_dim_pred + loss_rk + loss_overall

        return loss
    
def compute_metrics(eval_pred):

    logits, labels = eval_pred
    logits = torch.tensor(logits)
    rk_labels = labels.pop("rk_labels")
    w_labels = labels.pop("w_labels") 
    rk_labels, w_labels = torch.tensor(rk_labels), torch.tensor(w_labels)

    # Assuming model outputs logits in a format similar to what was described earlier
    scores, dim_prob, weights = torch.split(logits, logits.size(-1)//3, dim=-1)
    assert scores.size(0) % 3 == 0

    dim_prob_clone = dim_prob.clone()
    dim_prob_clone[1::3] = dim_prob[0::3]
    dim_prob_clone[2::3] = dim_prob[0::3]
    dim_prob = dim_prob_clone

    dim_prob = dim_prob.float()   # no sigmoid here
    correct_dim_prob = torch.where(
        (w_labels[0::3] != -100), ((dim_prob[0::3] >= 0.0) == (w_labels[0::3] >= 0.5)).float(), torch.tensor(float('nan'))
    )
    correct_dim_prob_acc = correct_dim_prob.nanmean()

    # scores
    scores = scores.sigmoid()
    # the range of scores now is (0, 1), weights * scores ∈ (0, 1)
    dim_prob = torch.where(dim_prob > 0.0, 1.0, 0.0)
    masked_weights = torch.where(dim_prob.bool(), weights, torch.tensor(float('-inf'), device=weights.device))
    masked_weights = masked_weights.softmax(dim=-1)
    scores = torch.cat((scores, (masked_weights * scores).sum(dim=-1).unsqueeze(-1)), dim=-1)


    logit_diffs = scores[1::3] - scores[2::3]

    valid_mask = (rk_labels[:, :-1] != -100) & (rk_labels[:, :-1] < 2.0)
    correct_scores = torch.where(
        valid_mask, ((logit_diffs[:, :-1] >= 0.0) == (rk_labels[:, :-1] >= 0.5)).float(), torch.tensor(float('nan'))
    )
    correct_scores_acc = correct_scores.nanmean()

    valid_mask = (rk_labels[:, -1:] != -100) & (rk_labels[:, -1:] < 2.0)
    correct_final = torch.where(
        valid_mask, ((logit_diffs[:, -1:] >= 0.0) == (rk_labels[:, -1:] >= 0.5)).float(), torch.tensor(float('nan'))
    )
    correct_final_acc = correct_final.nanmean()

    metrics = {"rank_acc": correct_scores_acc, "weight_acc": correct_dim_prob_acc, 'preference_acc': correct_final_acc}

    return metrics


def load_datasets(tokenizer, dataset_paths, eval_split_size, seed, label_field, num_workers, cache_dir):

    def preprocess_function(examples):
        results = {key: [] for key in label_field + ['instruction', 'dialogue_a', 'dialogue_b']}
        for prompt, response_a, response_b, annotation, scenario in zip(examples['prompt'], examples['response_a'], examples['response_b'], examples['annotation'], examples['scenario']):
            results['dialogue_a'].append(tokenizer.apply_chat_template([
                {'role': 'user', 'content': prompt if prompt is not None else ''},
                {'role': 'assistant', 'content': response_a if response_a is not None else ''},
            ], tokenize=False))
            results['dialogue_b'].append(tokenizer.apply_chat_template([
                {'role': 'user', 'content': prompt if prompt is not None else ''},
                {'role': 'assistant', 'content': response_b if response_b is not None else ''},
            ], tokenize=False))

            results['instruction'].append(tokenizer.apply_chat_template([{'role': 'user', 'content': prompt if prompt is not None else ''}], tokenize=False))

            for k in annotation.keys():
                assert k in label_field, f"{k} not in label_field"
            for k in label_field:
                if k not in annotation.keys() or annotation[k] is None:
                    assert WEIGHT_DICT[scenario]['attributes'].count(k) == 0, f"scenario {scenario} has {k} in attributes which is incorrectly set"
                    results[k].append([[-100, -100], [-100, -100]])

                elif k in annotation.keys() and not isinstance(annotation[k], dict):
                    if annotation[k] == '1':
                        results[k].append([[-100, 1], [0, -100]])
                    elif annotation[k] == 'tie':
                        results[k].append([[-100, 2], [2, -100]])
                    else:
                        results[k].append([[-100, 0], [1, -100]])
                else:
                    if annotation[k]['winner'] == '1':
                        results[k].append([[-100, 1], [0, -100]])
                    elif annotation[k]['winner'] == 'tie':
                        results[k].append([[-100, 2], [2, -100]])
                    else:
                        results[k].append([[-100, 0], [1, -100]])

        return results

    train_datasets = {}
    eval_datasets = {}

    loaded_datasets = {}
    for path in dataset_paths:
        try:
            dataset = datasets.load_from_disk(cache_dir)
        except:
            dataset = datasets.load_dataset("json", data_files=path, cache_dir=cache_dir)
        
        # reset the column names
        for split, data in dataset.items():
            dataset[split] = data.map(
                preprocess_function,
                batched=True,
                num_proc=num_workers,
                remove_columns=["prompt", "response_a", "response_b", "annotation"],
                keep_in_memory=True,
                desc="preprocessing new columns on dataset",
            )

        if isinstance(dataset, datasets.DatasetDict):
            if "train" in dataset and len(dataset) == 1:
                loaded_datasets[path] = dataset["train"]
            else:
                for split, ds in dataset.items():
                    loaded_datasets[f"{path}/{split}"] = ds
        else:
            loaded_datasets[path] = dataset

    for path, dataset in loaded_datasets.items():
        dataset = dataset.filter(LabelFilter(label_field), num_proc=num_workers, keep_in_memory=True)

        if eval_split_size < 1.0:
            splits = dataset.train_test_split(test_size=eval_split_size, seed=seed)
            train_dataset, eval_dataset = splits["train"], splits["test"]
        else:
            eval_dataset = dataset
            train_dataset = dataset.select([])

        dataset_name = os.path.basename(path)

        if dataset_name in train_datasets:
            train_datasets[dataset_name] = (
                datasets.concatenate_datasets([train_datasets[dataset_name], train_dataset])
            )
            eval_datasets[dataset_name] = (
                datasets.concatenate_datasets([eval_datasets[dataset_name], eval_dataset])
            )
        else:
            train_datasets[dataset_name] = train_dataset
            eval_datasets[dataset_name] = eval_dataset
    return train_datasets, eval_datasets


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
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
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
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
    )
    if args.config_overrides:
        logger.info(f"Overriding config: {args.config_overrides}")
        config.update_from_string(args.config_overrides)
        logger.info(f"New config: {config}")

    config.num_labels = 2 * (len(args.label_field) - 1)   # remove "Overall winner"
    tokenizer.pad_token_id = 0
    config.pad_token_id = 0

    if args.model_name_or_path:
        half_dtype = (torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None))
        device_map = {"":int(os.environ.get("LOCAL_RANK") or 0)}
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            config=config,
            cache_dir=args.cache_dir,
            revision=args.model_revision,
            use_auth_token=True if args.use_auth_token else None,
            torch_dtype=half_dtype,
            # torch_dtype=(half_dtype if args.lora or args.lora_path else None),
            # low_cpu_mem_usage=True,
            # device_map=device_map,
            use_flash_attention_2=training_args.use_flash_attention_2,
        )

        # # freeze all parameters
        for param in model.model.parameters():
            param.requires_grad = False
        
        in_features, out_features = model.score.in_features, model.score.out_features
        model.score = SaMerClassifier(input_dim=in_features, output_dim=out_features//2)
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
        
        # for name, param in model.score.named_parameters():
        #     param.requires_grad = True
        #     print(name, param.requires_grad)
        
        model.print_trainable_parameters()

    logger.info(f"Model: {model}")

    train_dataset = {}
    val_dataset = {}
    
    train_paths = Path(args.train_datasets_dir)
    train_files = [os.path.join(train_paths, f) for f in train_paths.glob("*.json")]

    train_dataset, val_dataset = load_datasets(
        tokenizer,
        train_files,    # list of json files
        args.eval_split_size_train if args.eval_split_size_train is not None else args.eval_split_size,
        training_args.seed,
        args.label_field,
        training_args.dataloader_num_workers,
        cache_dir=args.cache_dir
    )
    train_dataset = datasets.concatenate_datasets(list(train_dataset.values()))
    val_dataset = datasets.concatenate_datasets(list(val_dataset.values()))

    logger.warning(f"Before confidence filtering - train sequences: {len(train_dataset):,} - validation sequences: {len(val_dataset):,}")
    train_dataset = train_dataset.filter(
        ConfidenceFilter(args.label_field, training_args.confidence_threshold),
        num_proc=training_args.dataloader_num_workers, keep_in_memory=True
    )
    val_dataset = val_dataset.filter(
        ConfidenceFilter(args.label_field, training_args.confidence_threshold),
        num_proc=training_args.dataloader_num_workers, keep_in_memory=True
    )
    logger.warning(f"After confidence filtering - train sequences: {len(train_dataset):,} - validation sequences: {len(val_dataset):,}")

    if training_args.do_eval:
        eval_paths = Path(args.eval_datasets_dir)
        eval_files = [os.path.join(eval_paths, f) for f in eval_paths.glob("*.json")]

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

    collator = DataCollator(args, training_args, tokenizer)

    # Initialize our Trainer
    trainer = PreferenceTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    if trainer.is_fsdp_enabled:
        # Identify which modules have "layer" in their class name and use these
        # as the basic FSDP blocks that are sharded and exchanged between GPUs
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
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
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