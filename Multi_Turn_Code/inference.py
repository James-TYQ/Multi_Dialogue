import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, LlamaForSequenceClassification
from typing import Dict, List
import logging

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

class SaMerPipeline:
    def __init__(self, model_id, trust_remote_code=True, torch_dtype=torch.float32, device_map=None, max_length=4096):
        """
        初始化SaMer评估管道
        
        Args:
            model_id: 模型ID或本地路径
            trust_remote_code: 是否信任远程代码
            torch_dtype: 模型使用的数据类型
            device_map: 设备映射
            max_length: 最大序列长度
        """
        self.model_id = model_id
        self.torch_dtype = torch_dtype
        self.max_length = max_length
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            trust_remote_code=trust_remote_code,
            use_fast=True
        )
        
        # 确保tokenizer有pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 修改：使用更安全的模型加载方式，避免device_map和内存问题
        try:
            # 尝试使用CPU加载模型，避免OOM
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                offload_folder="offload_folder",  # 添加离线文件夹
                offload_state_dict=True,  # 启用状态字典卸载
                device_map=None  # 明确设置为None
            )
            logger.info("成功使用AutoModelForSequenceClassification加载模型")
        except Exception as e:
            logger.warning(f"使用AutoModelForSequenceClassification加载失败: {e}")
            try:
                # 尝试使用LlamaForSequenceClassification加载
                self.model = LlamaForSequenceClassification.from_pretrained(
                    model_id,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    offload_folder="offload_folder",
                    offload_state_dict=True,
                    device_map=None
                )
                logger.info("成功使用LlamaForSequenceClassification加载模型")
            except Exception as e2:
                logger.error(f"使用LlamaForSequenceClassification加载也失败: {e2}")
                
                # 最后尝试：使用safetensors加载
                try:
                    from transformers import AutoConfig
                    config = AutoConfig.from_pretrained(model_id)
                    
                    # 根据配置创建模型
                    if hasattr(config, "model_type") and config.model_type == "llama":
                        self.model = LlamaForSequenceClassification.from_pretrained(
                            model_id,
                            config=config,
                            trust_remote_code=trust_remote_code,
                            torch_dtype=torch_dtype,
                            low_cpu_mem_usage=True,
                            device_map=None
                        )
                    else:
                        self.model = AutoModelForSequenceClassification.from_pretrained(
                            model_id,
                            config=config,
                            trust_remote_code=trust_remote_code,
                            torch_dtype=torch_dtype,
                            low_cpu_mem_usage=True,
                            device_map=None
                        )
                    logger.info("成功使用配置方式加载模型")
                except Exception as e3:
                    logger.error(f"所有加载方式都失败: {e3}")
                    raise RuntimeError(f"无法加载模型: {e}, {e2}, {e3}")
        
        # 加载敏感度和特异度参数
        self.sensitivity = 0.8
        self.specificity = 0.6
        
        # 尝试加载训练好的敏感度和特异度参数
        ss_path = os.path.join(model_id, "sensitivity_specificity.pt")
        if os.path.exists(ss_path):
            try:
                ss_dict = torch.load(ss_path, map_location="cpu")
                self.sensitivity = ss_dict.get("sensitivity_value", 0.8)
                self.specificity = ss_dict.get("specificity_value", 0.6)
                logger.info(f"加载敏感度: {self.sensitivity:.4f}, 特异度: {self.specificity:.4f}")
            except Exception as e:
                logger.warning(f"加载敏感度和特异度参数失败: {e}")
        
        # 将模型移动到GPU（如果可用）
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 修改：使用to方法时添加非阻塞选项，避免OOM
        if self.device == "cuda":
            try:
                # 尝试将整个模型移到GPU
                self.model = self.model.to(self.device)
            except Exception as e:
                logger.warning(f"无法将整个模型加载到GPU: {e}")
                self.device = "cpu"
                logger.info("回退到CPU模式")
        
        self.model.eval()
        self.dimensions = ["结论"]
    
    # 其余方法保持不变
    def __call__(self, messages: List[Dict[str, str]]) -> Dict:
        """
        评估对话质量
        
        Args:
            messages: 对话消息列表，每个消息是一个字典，包含role和content
            
        Returns:
            包含评估结果的字典
        """
        # 构建输入文本
        input_text = self._format_messages(messages)
        
        # 对输入文本进行编码
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        # 进行推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        conclusion_score = torch.sigmoid(logits[0, 0]).item()
        dimensional_scores = {"结论": conclusion_score}
        overall_score = conclusion_score
        evaluation_dim = ["结论"]
        
        return {
            "dimensional_score": dimensional_scores,
            "overall_score": overall_score,
            "evaluation_dim": evaluation_dim,
            "conclusion": dimensional_scores.get("结论", 0.5)
        }
    
    def _format_messages(self, messages):
        """
        将消息列表格式化为模型输入文本
        
        Args:
            messages: 对话消息列表
            
        Returns:
            格式化后的文本
        """
        formatted_text = ""
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "user":
                formatted_text += f"Human: {content}\n\n"
            elif role == "assistant":
                formatted_text += f"Assistant: {content}\n\n"
        
        return formatted_text.strip()
    
    def compare_responses(self, user_query, response_a, response_b):
        """
        比较两个回答的质量
        
        Args:
            user_query: 用户问题
            response_a: 回答A
            response_b: 回答B
            
        Returns:
            比较结果，包括胜者和各自的得分
        """
        # 构建对话A
        dialog_a = [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": response_a}
        ]
        
        # 构建对话B
        dialog_b = [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": response_b}
        ]
        
        # 评估对话A
        result_a = self(dialog_a)
        
        # 评估对话B
        result_b = self(dialog_b)
        
        # 比较得分
        score_a = result_a["overall_score"]
        score_b = result_b["overall_score"]
        
        if abs(score_a - score_b) < 0.1:
            winner = "Fair"
        elif score_a > score_b:
            winner = "A"
        else:
            winner = "B"
        
        return {
            "winner": winner,
            "score_a": score_a,
            "score_b": score_b,
            "dims_a": result_a["dimensional_score"],
            "dims_b": result_b["dimensional_score"]
        }