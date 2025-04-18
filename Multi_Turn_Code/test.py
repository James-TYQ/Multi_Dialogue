import os
import csv
import torch
import pandas as pd
from tqdm import tqdm
import logging
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_log.txt')
    ]
)
logger = logging.getLogger(__name__)

# 添加 token 认证
from huggingface_hub import login
login(token="hf_xxx")

# 模型路径
MODEL_PATH = "./results/multi_turn_ArmoRM-Llama3-8B-v0.1_bsz16_lr5e-5_ss_lr5e-3_sens0.8_spec0.6_epochs1_warmup0.1_conf0.0_labeltemp2.0multi-turn-dialogue"
CSV_PATH = "e:/SaMer/SaMer/Multi_Turn_Train_Data/test/Sample.csv"

# 确保inference模块可以被导入
try:
    from inference import SaMerPipeline
    logger.info("成功导入SaMerPipeline")
except ImportError as e:
    logger.error(f"导入SaMerPipeline失败: {e}")
    sys.exit(1)

# 初始化评估管道
logger.info(f"正在加载模型: {MODEL_PATH}")
try:
    pipeline = SaMerPipeline(
        model_id=MODEL_PATH,  # 使用本地训练好的模型
        trust_remote_code=True,
        torch_dtype=torch.float32  # 使用 float32 避免精度问题
    )
    logger.info("模型加载完成")
except Exception as e:
    logger.error(f"加载模型失败: {e}", exc_info=True)
    sys.exit(1)

def parse_conversation(conversation_text):
    """解析对话文本，提取人类和助手A/B的对话"""
    lines = conversation_text.strip().split('\n\n')
    dialog = []
    
    i = 0
    while i < len(lines):
        # 处理人类消息
        if '"role": "Human"' in lines[i]:
            human_text = lines[i+1].strip().strip('"')
            if human_text.startswith("text"):
                human_text = human_text[7:].strip().strip('"')
            dialog.append({"role": "user", "content": human_text})
            i += 2
        # 处理助手消息
        elif '"role": "Assistant"' in lines[i]:
            a_text = lines[i+1].strip().strip('"')
            if a_text.startswith("A"):
                a_text = a_text[3:].strip().strip('"')
            
            b_text = lines[i+2].strip().strip('"')
            if b_text.startswith("B"):
                b_text = b_text[3:].strip().strip('"')
            
            dialog.append({"role": "assistant_A", "content": a_text})
            dialog.append({"role": "assistant_B", "content": b_text})
            i += 3
        else:
            i += 1
    
    return dialog

def evaluate_conversation(dialog):
    """评估对话中A和B的回答质量"""
    # 构建A的对话
    dialog_A = []
    # 构建B的对话
    dialog_B = []
    
    for i, msg in enumerate(dialog):
        if msg["role"] == "user":
            dialog_A.append({"role": "user", "content": msg["content"]})
            dialog_B.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "assistant_A":
            dialog_A.append({"role": "assistant", "content": msg["content"]})
        elif msg["role"] == "assistant_B":
            dialog_B.append({"role": "assistant", "content": msg["content"]})
    
    # 评估A的对话
    try:
        result_A = pipeline(dialog_A)
        # 评估B的对话
        result_B = pipeline(dialog_B)
    except Exception as e:
        logger.error(f"评估对话时出错: {e}", exc_info=True)
        raise
    
    # 比较A和B的总体得分
    score_A = result_A['overall_score']
    score_B = result_B['overall_score']
    
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
        "A_conclusion": result_A.get('conclusion', score_A),
        "B_conclusion": result_B.get('conclusion', score_B)
    }

def main():
    # 读取CSV文件
    logger.info(f"正在读取CSV文件: {CSV_PATH}")
    try:
        df = pd.read_csv(CSV_PATH)
        logger.info(f"共读取 {len(df)} 条对话")
    except Exception as e:
        logger.error(f"读取CSV文件失败: {e}")
        return
    
    # 准备结果存储
    results = []
    
    # 处理每个对话
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="评估对话"):
        conversation_text = row['conversation']
        dialog = parse_conversation(conversation_text)
        
        # 跳过空对话
        if not dialog:
            logger.warning(f"对话 {idx} 解析失败，跳过")
            continue
        
        # 评估对话
        try:
            eval_result = evaluate_conversation(dialog)
            
            # 获取真实标签
            ground_truth = row.get('out_conclusion', 'Unknown')
            
            # 保存结果
            results.append({
                "dialog_id": row.get('dialog_id', idx),
                "A_score": eval_result["A_score"],
                "B_score": eval_result["B_score"],
                "model_winner": eval_result["winner"],
                "ground_truth": ground_truth,
                "correct": eval_result["winner"] == ground_truth,
                "A_conclusion": eval_result.get("A_conclusion", eval_result["A_score"]),
                "B_conclusion": eval_result.get("B_conclusion", eval_result["B_score"])
            })
            
            # 打印当前结果
            logger.info(f"\n对话 {idx}:")
            logger.info(f"模型判断: {eval_result['winner']}, 真实标签: {ground_truth}")
            logger.info(f"A得分: {eval_result['A_score']:.4f}, B得分: {eval_result['B_score']:.4f}")
            if "A_conclusion" in eval_result:
                logger.info(f"A结论得分: {eval_result['A_conclusion']:.4f}")
                logger.info(f"B结论得分: {eval_result['B_conclusion']:.4f}")
            
        except Exception as e:
            logger.error(f"评估对话 {idx} 时出错: {e}", exc_info=True)
    
    # 计算准确率
    if results:
        correct_count = sum(1 for r in results if r["correct"])
        accuracy = correct_count / len(results) if results else 0
        
        # 保存结果到CSV
        result_df = pd.DataFrame(results)
        result_path = os.path.join(os.path.dirname(CSV_PATH), "evaluation_results.csv")
        result_df.to_csv(result_path, index=False)
        
        logger.info(f"\n评估完成! 总共评估 {len(results)} 个对话")
        logger.info(f"准确率: {accuracy:.4f} ({correct_count}/{len(results)})")
        logger.info(f"详细结果已保存到: {result_path}")
    else:
        logger.warning("没有成功评估任何对话")

if __name__ == "__main__":
    main()