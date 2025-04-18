import json
import re

# 读取 JSON 文件
with open('all_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 定义需要检查的维度
required_dimensions = [
    "准确性", "逻辑性", "口语性", "关联性", "个性化", 
    "创造性", "互动性", "情感性", "知识性", "安全性", "结论"
]

# 定义有效的评分
valid_scores = ["A", "B", "Fair"]

# 用于存储不符合规范的评估
invalid_evaluations = []

# 检查每个对话的评估
for dialog in data:
    dialog_id = dialog.get("dialog_id", "未知")
    
    if "evaluations" not in dialog:
        continue
    
    for eval_item in dialog["evaluations"]:
        annotator = eval_item.get("annotator", "未知")
        evaluation_text = eval_item.get("evaluation", "")
        
        # 尝试解析评估文本为 JSON
        try:
            # 检查是否是 JSON 格式的字符串
            if not (evaluation_text.strip().startswith("{") and evaluation_text.strip().endswith("}")):
                invalid_evaluations.append(f"{dialog_id}: {annotator}")
                continue
                
            # 尝试解析 JSON
            evaluation_dict = json.loads(evaluation_text)
            
            # 检查是否包含所有必要的维度
            missing_dimensions = [dim for dim in required_dimensions if dim not in evaluation_dict]
            if missing_dimensions:
                invalid_evaluations.append(f"{dialog_id}: {annotator} (缺少维度: {', '.join(missing_dimensions)})")
                continue
            
            # 检查每个维度的评分是否有效
            invalid_scores = [f"{dim}: {evaluation_dict[dim]}" for dim in required_dimensions 
                             if dim in evaluation_dict and evaluation_dict[dim] not in valid_scores]
            if invalid_scores:
                invalid_evaluations.append(f"{dialog_id}: {annotator} (无效评分: {', '.join(invalid_scores)})")
                continue
                
        except json.JSONDecodeError:
            # 如果不是有效的 JSON 格式
            invalid_evaluations.append(f"{dialog_id}: {annotator}")
            continue

# 将结果写入文本文件
with open('invalid_evaluations.txt', 'w', encoding='utf-8') as f:
    f.write("不符合规范的评估:\n")
    for item in invalid_evaluations:
        f.write(f"{item}\n")

print(f"检查完成，共发现 {len(invalid_evaluations)} 个不符合规范的评估。")
print(f"结果已保存到 invalid_evaluations.txt")