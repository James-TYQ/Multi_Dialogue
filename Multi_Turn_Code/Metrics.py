import torch

# Use for Evaluation
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
                    a_wins = (valid_labels == 0).sum().item()   
                    b_wins = (valid_labels == 1).sum().item()            # 计算A胜(0)和B胜(1)的数量

                    if a_wins - b_wins >= 3:
                        processed_labels[i] = 0  # A胜
                    elif b_wins - a_wins >= 3:
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