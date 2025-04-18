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
        self.label_field = label_field if label_field else ["结论"]
        
        init_sens = torch.logit(torch.tensor(self.args.initial_sensitivity,
                                             device=self.args.device))
        init_spec = torch.logit(torch.tensor(self.args.initial_specificity,
                                             device=self.args.device))

        self.sensitivity = nn.Parameter(init_sens)   # leaf, requires_grad=True
        self.specificity = nn.Parameter(init_spec)
        
        logger.info(f"初始敏感度: {init_sens}, 初始特异度: {init_spec}")
        
        # 将参数移动到正确的设备上
        self._move_model_to_device(self.model, self.args.device)
        self.sensitivity.data = self.sensitivity.data.to(self.args.device)
        self.specificity.data = self.specificity.data.to(self.args.device)

    def create_optimizer(self):
        """创建包含敏感度和特异度参数的优化器"""
        if self.optimizer is None:
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            optimizer_grouped_parameters = [
                {"params": trainable, "weight_decay": self.args.weight_decay},
                {"params": [self.sensitivity, self.specificity],
                "weight_decay": 0.0,
                "lr": self.args.ss_learning_rate},
            ]

            
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        
        return self.optimizer
        
    def compute_loss(self, model, inputs, return_outputs=False):
        # 获取标签和模型输出
        labels = inputs.pop("labels")
        outputs = model(**inputs, use_cache=False)
        
        # 处理不同类型的输出
        if isinstance(outputs, tuple):
            logits = outputs[0]
            var = outputs[1] if len(outputs) > 1 else torch.ones_like(logits) * 1e-6
        else:
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            var = torch.ones_like(logits) * 1e-6
        
        # 计算分数差异
        score_diff = logits[0::2] - logits[1::2]
        
        # 初始化总损失
        total_loss = 0.0
        metrics = {}
        
        # 处理每个维度的标签
        for conclusion_dim in self.label_field:
            if conclusion_dim in labels:
                dim_labels = labels[conclusion_dim]
                
                # 计算概率
                score_var = var[0::2]**2 + var[1::2]**2 + 1e-4
                p = 0.5 * (1 + torch.erf(-score_diff / torch.sqrt(2 * score_var)))
                
                # 使用sigmoid获取当前的sensitivity和specificity值
                sensitivity = torch.sigmoid(self.sensitivity)
                specificity = torch.sigmoid(self.specificity)
                
                alpha = torch.stack((1 - sensitivity, sensitivity)).to(p.device)      
                beta  = torch.stack((specificity, 1 - specificity)).to(p.device)
                
                # 计算损失和准确率
                loss, sample_losses = self.binary_loss(p, dim_labels, alpha=alpha, beta=beta)
                
                # 计算准确率
                valid_mask = (dim_labels == 0) | (dim_labels == 1)
                if valid_mask.sum() > 0:
                   pred = (p > 0.5).float()
                   accuracy = (pred == dim_labels.float()).float()[valid_mask].mean()
                else:
                    accuracy = torch.tensor(0.0, device=p.device)
                
                total_loss += loss
                
                # 记录指标
                metrics[f"{conclusion_dim}_acc"] = accuracy.item()
                metrics["sensitivity"] = sensitivity.item()
                metrics["specificity"] = specificity.item()
                
                self.log(metrics)
        
        if return_outputs:
            if self.model.training:  # 只在训练阶段执行调试
                loss.backward()
                print(">>> grad_sens:", self.sensitivity.grad)
                print(">>> grad_spec:", self.specificity.grad)
            return total_loss, outputs
        
        return total_loss
        
    def save_model(self, output_dir=None, _internal_call=False):
        """保存模型和敏感度/特异度参数"""
        # 调用父类的save_model方法保存模型
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        
        # 保存敏感度和特异度参数
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 计算当前的敏感度和特异度值
        sensitivity_value = torch.sigmoid(self.sensitivity).item()
        specificity_value = torch.sigmoid(self.specificity).item()
        
        # 创建包含参数和值的字典
        ss_dict = {
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "sensitivity_value": sensitivity_value,
            "specificity_value": specificity_value
        }
        
        # 保存参数
        torch.save(ss_dict, os.path.join(output_dir, "sensitivity_specificity.pt"))
        logger.info(f"保存敏感度: {sensitivity_value:.4f}, 特异度: {specificity_value:.4f}")
        
    def _load_from_checkpoint(self, resume_from_checkpoint):
        """从检查点加载模型和敏感度/特异度参数"""
        # 调用父类的方法加载模型
        super()._load_from_checkpoint(resume_from_checkpoint)
        
        # 加载敏感度和特异度参数
        ss_path = os.path.join(resume_from_checkpoint, "sensitivity_specificity.pt")
        if os.path.exists(ss_path):
            ss_dict = torch.load(ss_path, map_location="cpu")

            # 用 copy_ / data 方式把权重拷进来，而不是重新赋值
            self.sensitivity.data.copy_(ss_dict["sensitivity"].data)
            self.specificity.data.copy_(ss_dict["specificity"].data)

            # 若设备不同，再原地转移
            self.sensitivity.data = self.sensitivity.data.to(self.args.device)
            self.specificity.data = self.specificity.data.to(self.args.device)

            logger.info("加载敏感度: %.4f, 特异度: %.4f",
                        torch.sigmoid(self.sensitivity).item(),
                        torch.sigmoid(self.specificity).item())
        else:
            logger.warning(f"未找到敏感度和特异度参数文件: {ss_path}")