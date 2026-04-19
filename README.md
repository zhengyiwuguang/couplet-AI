# 智能对联生成器
基于Transformer的中文对联自动生成系统，支持平仄约束、对仗生成

## 环境
Python 3.8+
PyTorch 1.12+

## 安装依赖
pip install -r requirements.txt

## 数据集
couplet.txt：7万对联对，格式：上联\t下联

## 训练
python train.py

## 推理
python generate.py

## 功能
输入上联 → 自动生成合规下联