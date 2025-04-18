from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification, LlamaForSequenceClassification
import torch
import torch.nn as nn
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

class LlamaForSaMerModel(LlamaForSequenceClassification):
    """使用SaMer分类器的Llama模型"""
    def __init__(self, config):
        super().__init__(config)
        self.score = SaMerClassifier(config.hidden_size, config.num_labels)

class SaMerPipeline:
    """多轮对话评估管道"""
    def __init__(self, model_id, device_map="auto", torch_dtype=torch.bfloat16, 
                 truncation=True, trust_remote_code=False, max_length=8192):
        """
        初始化SaMer评估管道
        
        Args:
            model_id: 模型ID或本地路径
            device_map: 设备映射策略
            torch_dtype: 模型使用的数据类型
            truncation: 是否截断过长的输入
            trust_remote_code: 是否信任远程代码
            max_length: 最大序列长度
        """
        # 加载模型和分词器
        try:
            self.model = LlamaForSaMerModel.from_pretrained(
                model_id,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype,
            )
            logger.info("成功加载LlamaForSaMerModel模型")
        except Exception as e:
            logger.warning(f"加载LlamaForSaMerModel失败: {e}")
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_id,
                    device_map=device_map,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=torch_dtype,
                )
                logger.info("成功加载AutoModelForSequenceClassification模型")
            except Exception as e:
                logger.error(f"加载模型失败: {e}")
                raise
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        
        # 确保tokenizer有pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, float]:
        """
        评估对话质量
        
        Args:
            messages: 对话消息列表，每个消息是一个字典，包含role和content
            
        Returns:
            包含评估结果的字典
        """
        # 构建输入文本
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        input_ids = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length
        ).to(self.device)
        
        # 进行推理
        with torch.no_grad():
            outputs = self.model(**input_ids)
            logits = outputs.logits
        
        # 计算结论得分
        conclusion_score = torch.sigmoid(logits[0, 0]).item()
        
        return {
            "dimensional_score": {"结论": conclusion_score},
            "overall_score": conclusion_score,
            "evaluation_dim": ["结论"],
            "conclusion": conclusion_score
        }
    
    def compare_responses(self, dialog_A, dialog_B):
        """
        比较两个对话的质量
        
        Args:
            dialog_A: 对话A的消息列表
            dialog_B: 对话B的消息列表
            
        Returns:
            比较结果，包括胜者和各自的得分
        """
        result_A = self(dialog_A)
        result_B = self(dialog_B)
        
        score_A = result_A["overall_score"]
        score_B = result_B["overall_score"]
        
        # 确定胜者
        if abs(score_A - score_B) < 0.1:
            winner = "Fair"
        elif score_A > score_B:
            winner = "A"
        else:
            winner = "B"
        
        return {
            "A_score": score_A,
            "B_score": score_B,
            "winner": winner,
            "A_conclusion": result_A["conclusion"],
            "B_conclusion": result_B["conclusion"]
        }