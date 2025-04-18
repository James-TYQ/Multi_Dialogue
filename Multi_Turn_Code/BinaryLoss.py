import torch
import torch.nn as nn

class BinaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, probs, labels, alpha, beta):
        # 处理维度不匹配的情况
        if len(probs.shape) == 1:
            probs = probs.unsqueeze(1)  # 变成 [batch_size, 1]
        if len(labels.shape) == 1:
            labels = labels.unsqueeze(1)  # 变成 [batch_size, 1]
        
        # 处理多标注者情况
        if probs.shape[1] == 1 and labels.shape[1] > 1:
            probs = probs.expand(-1, labels.shape[1])
        
        valid_mask = (labels == 0) | (labels == 1)
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=probs.device), torch.tensor(0.0, device=probs.device)

        # 只保留有效标签样本
        valid_probs = probs[valid_mask]
        valid_labels = labels[valid_mask].float()
        
        # 确保标签值在[0,1]范围内
        valid_labels = torch.clamp(valid_labels, 0, 1)
        
        # 使用Binary_Loss的方法计算损失
        g = torch.zeros((valid_labels.size(0), 2), device=valid_labels.device)
        g.scatter_(1, valid_labels.long().unsqueeze(1), 1)
        
        # 将预测概率转换为二分类格式
        p_binary = torch.stack([1-valid_probs, valid_probs], dim=1)
        
        # 计算对数似然
        log_a = g * torch.log(alpha) + (1 - g) * torch.log(1 - alpha)
        log_a = torch.sum(log_a, dim=1, keepdim=True)
        a = torch.exp(log_a)

        log_b = (1 - g) * torch.log(beta) + g * torch.log(1 - beta)
        log_b = torch.sum(log_b, dim=1, keepdim=True)
        b = torch.exp(log_b)
        
        p_sum = torch.sum(p_binary * g, dim=1, keepdim=True)
        loss_val = torch.log(a * p_sum + b * (1 - p_sum)) / 2
        
        # 计算准确率
        accuracy = ((valid_probs > 0.5).float() == valid_labels).float().mean() if valid_mask.sum() > 0 else torch.tensor(0.0, device=probs.device)
        
        return -torch.mean(loss_val), loss_val  # 返回平均损失和每个样本的损失