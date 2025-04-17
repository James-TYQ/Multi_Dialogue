import os
import torch
import torch.nn as nn
import logging
from transformers import Trainer

from BinaryLoss import BinaryLoss

logger = logging.getLogger(__name__)

class PreferenceTrainer(Trainer):
    def __init__(self, *args, label_field=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.binary_loss = BinaryLoss()
        
        # 保存label_field属性
        self.label_field = label_field
        
        # 创建可训练的敏感度和特异度参数，使用args中的初始值, 将0-1范围的值转换为logit值
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
            lr=self.args.learning_rate,
            weight_decay=5e-4,
        )
        
        logger.info(f"初始敏感度: {self.args.initial_sensitivity}, 初始特异度: {self.args.initial_specificity}")

    def compute_loss(self, model, inputs, return_outputs=False, return_output_and_metrics=False):
        # 获取标签和模型输出
        labels = inputs.pop("labels")
        outputs = model(**inputs, use_cache=False)
        
        if isinstance(outputs, tuple):
            logits = outputs[0]         # 如果outputs是元组，假设第一个元素是logits
        else:
            logits = outputs.logits     # 否则直接使用logits
        
        print("logits:", logits)
        device = logits.device
        
        # 计算A和B回答的评分差异
        score_diff = logits[0::2] - logits[1::2]             # 质量评估损失 (loss_q)
        total_loss = torch.tensor(0.0, device=device)        # 初始化总损失为0
        
        conclusion_dim = "结论"
        if conclusion_dim in labels:
            dim_labels = labels[conclusion_dim]
            
            # 使用固定方差为1
            score_var = torch.ones_like(score_diff)
            p = 0.5 * (1 + torch.erf(-score_diff / torch.sqrt(2 * score_var)))
            
            if len(p.shape) == 1:
                p = p.unsqueeze(1)
            
            print("p:", p)
            print("dim_labels:", dim_labels)

            if p.shape[0] != dim_labels.shape[0]:
                if p.shape[0] == 1 and dim_labels.shape[0] > 1:
                    p = p.expand(dim_labels.shape[0], -1)
                elif dim_labels.shape[0] == 1 and p.shape[0] > 1:
                    p = p[0].unsqueeze(0)
            
            # 创建alpha和beta张量
            alpha = torch.tensor([1-torch.sigmoid(self.sensitivity), torch.sigmoid(self.sensitivity)], device=p.device)
            beta = torch.tensor([torch.sigmoid(self.specificity), 1-torch.sigmoid(self.specificity)], device=p.device)
            
            # 直接传递alpha和beta
            loss, _ = self.binary_loss(p, dim_labels, alpha=alpha, beta=beta)
            total_loss += loss
            
            # 计算准确率
            accuracy = ((p > 0.5).float() == dim_labels).float().mean()
            
            self.log({
                "loss": total_loss.item(),
                "accuracy": accuracy.item(),
                "sensitivity": torch.sigmoid(self.sensitivity).item(),
                "specificity": torch.sigmoid(self.specificity).item()
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
        
        os.makedirs(output_dir, exist_ok=True)              # 确保目录存在
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
        
        ss_path = os.path.join(resume_from_checkpoint, "sensitivity_specificity.pt")           # 尝试加载敏感度和特异度参数
        if os.path.exists(ss_path):
            ss_dict = torch.load(ss_path)
            self.sensitivity = ss_dict["sensitivity"]
            self.specificity = ss_dict["specificity"]
            logger.info(f"Loaded sensitivity: {torch.sigmoid(self.sensitivity).item()}, specificity: {torch.sigmoid(self.specificity).item()}")