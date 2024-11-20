from scipy.stats import pearsonr, spearmanr, kendalltau
import json
import numpy as np

f = open("./data/gpt-4o.json", "r", encoding="utf-8")
gpt_data = json.load(f)
f = open("./data/SaMer.json", "r", encoding="utf-8")
samer_data = json.load(f)

# check whether the data items are the same
assert len(gpt_data) == len(samer_data)
for d1, d2 in zip(gpt_data, samer_data):
    assert d1['instruction'] == d2['instruction']
    assert d1['ground_truth'] == d2['ground_truth']

ground_truth = [np.mean(d['ground_truth']) for d in gpt_data]
gpt_pred = [d['pred_score'] for d in gpt_data]
samer_pred = [d['pred_score'] for d in samer_data]

print("Vicuna Bench Results:")
print(f">>> GPT-4o <<<\nPearson Correlation: {pearsonr(ground_truth, gpt_pred)[0]:.4f}\nSpearman Correlation: {spearmanr(ground_truth, gpt_pred)[0]:.4f}\nKendall Correlation: {kendalltau(ground_truth, gpt_pred)[0]:.4f}")
print(f">>> SaMer <<<\nPearson Correlation: {pearsonr(ground_truth, samer_pred)[0]:.4f}\nSpearman Correlation: {spearmanr(ground_truth, samer_pred)[0]:.4f}\nKendall Correlation: {kendalltau(ground_truth, samer_pred)[0]:.4f}")