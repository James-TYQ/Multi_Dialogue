import os
import json
import torch
import datasets
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class LabelFilter:
    def __init__(self, label_field):
        self.label_field = label_field

    def __call__(self, example):
        for label in self.label_field:       # 只要有一个标签不是-100就保留
            if label in example and isinstance(example[label], torch.Tensor):
                if len(example[label]) > 0 and not (example[label] == -100).all().item():
                    return True
        return ('dialogue_a' in example and 'dialogue_b' in example and      # 如果至少有对话数据，也保留
                example['dialogue_a'] and example['dialogue_b'])

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