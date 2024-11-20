from typing import Dict, List
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn as nn


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

class SaMerPipeline:
    dimensions = ['Accuracy', 'Admit Uncertainty', 'Audience Friendly', 'Authenticity', 'Citation', 'Clarity', 'Coverage', 'Creativity', 'Depth', 'Feasibility', 'Harmlessness', 'Information Richness', 'Insight', 'Logic', 'Multiple Aspects', 'Objectivity', 'Originality', 'Professionalism', 'Relevance', 'Timeliness', 'Attractive', 'Interactivity', 'Professional', 'Being Friendly', 'Coherence', 'Emojis', 'Emotion', 'Length', 'Style', 'Vivid', 'Step by Step Explanation', 'Code Correctness', 'Code Readability', 'Instruction Following', 'Layout', 'Modularity', 'Pacing', 'Completeness', 'Faithfulness', 'Pointing Out', 'Result at the Beginning', 'Engagement']

    def __init__(self, model_id, device_map="auto", torch_dtype=torch.bfloat16, truncation=True, trust_remote_code=False, max_length=8192):
        # load model and tokenizer
        self.model = LlamaForMDQRwardModel.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, float]:
        """
        messages: OpenAI chat messages to be scored, i.e., [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
        return: a dictionary of results
        """
        assert len(messages) <= 2, 'message format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]'

        messages = [[{'role': 'user', 'content': messages[0][0]['content']}]] + messages

        # apply chat template
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

        with torch.no_grad():
            try:
                outputs = self.model(**input_ids)
                logits = outputs.logits
                scores, dim_prob, weights = torch.split(logits, logits.size(-1)//3, dim=-1)

                dim_prob_clone = dim_prob[0].unsqueeze(0).repeat(dim_prob.size(0), 1)
                dim_prob = dim_prob_clone

                dim_prob = torch.where(dim_prob > 0.0, 1.0, 0.0)
                masked_weights = torch.where(dim_prob.bool(), weights, torch.tensor(float('-inf'), device=weights.device))
                masked_weights = masked_weights.softmax(dim=-1)

                overall_scores = (masked_weights * scores.sigmoid()).sum(dim=-1)
                overall_scores = torch.tensor(overall_scores[1:].clone().detach(), dtype=torch.float32)

                selected_dims = [self.dimensions[i] for i, p in enumerate(dim_prob) if p == 1.0]
                
            except Exception as e:
                print(e)
                raise e

        return {
            "evaluation_dim": selected_dims,
            "dimensional_score": scores[1],
            "overall_score": overall_scores, 
        }