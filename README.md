Project Goal: Learning a light-weight multi-turn dialogue evaluator from multiple noisy annotators

Usage: samer虚拟环境

相关脚本: 1. train_Multi.sh (执行脚本)
         2. Multi_Turn_Train.py (训练脚本)
         3. Multi_Turn_Train_Data （文件夹存储数据），Multi_Turn_Train_Data/eval为测试集，Multi_Turn_Train_Data/train为部分训练集，Multi_Turn_Train_Data/temp为临时数据，all_data.json为当前所有训练数据

主要改动来自samer: 1. 只训练了一个scoring layer
                2. 利用Binary Loss作为质量损失函数
                3. 添加sensitivity和specificity作为训练需要权重

目前问题: 1. sensitibity和specificity参数无法更新
         2. 损失函数无法收敛 (似乎固定为ln2) 

