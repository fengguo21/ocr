# 身份证识别模型训练指南

## 📋 概述

本指南将带你完成CRNN（文本识别）和CRAFT（文本检测）两个模型的训练过程。

## 🎯 训练流程图

```
1. 准备训练数据
   ↓
2. 训练CRNN模型（文本识别）
   ↓  
3. 训练CRAFT模型（文本检测）
   ↓
4. 集成模型并测试
```

## 📁 文件结构

```
ocr/
├── train_crnn.py           # CRNN训练脚本
├── train_craft.py          # CRAFT训练脚本
├── prepare_training_data.py # 数据准备脚本
├── models/
│   ├── crnn.py            # CRNN模型定义
│   └── text_detection.py   # CRAFT模型定义
├── utils/
│   ├── image_processing.py
│   └── id_info_extractor.py
└── sample_training_data/   # 生成的训练数据
    ├── crnn_data/
    │   ├── images/         # 文本图像
    │   └── labels.txt      # 标签文件
    ├── craft_data/
    │   ├── img_*.jpg       # 场景图像
    │   └── gt_*.txt        # 标注文件
    └── dataset_info.json
```

## 🚀 步骤1: 准备训练数据

### 生成示例数据
```bash
# 生成CRNN训练数据
python prepare_training_data.py --model crnn --output training_data

# 生成CRAFT训练数据  
python prepare_training_data.py --model craft --output training_data

# 生成所有数据
python prepare_training_data.py --model both --output training_data
```

### 数据格式说明

**CRNN数据格式**:
```
crnn_data/
├── images/
│   ├── 00000.jpg    # 32px高度的文本图像
│   ├── 00001.jpg
│   └── ...
└── labels.txt       # 格式: image_name text
```

**CRAFT数据格式**:
```
craft_data/
├── img_000.jpg      # 完整场景图像
├── img_001.jpg
├── gt_img_000.txt   # 格式: x1,y1,x2,y2,x3,y3,x4,y4,text
└── gt_img_001.txt
```

## 🎯 步骤2: 训练CRNN模型（文本识别）

### 基础训练
```bash
# 使用默认参数训练
python train_crnn.py

# 指定参数训练
python train_crnn.py \
    --data_dir sample_training_data/crnn_data/images \
    --label_file sample_training_data/crnn_data/labels.txt \
    --batch_size 16 \
    --epochs 100 \
    --lr 0.001 \
    --save_dir checkpoints_crnn
```

### 训练参数说明
- `--data_dir`: 训练图像目录
- `--label_file`: 标签文件路径（支持txt和json格式）
- `--batch_size`: 批大小（推荐16-32）
- `--epochs`: 训练轮数
- `--lr`: 学习率
- `--save_dir`: 模型保存目录
- `--resume`: 恢复训练的检查点路径

### 恢复训练
```bash
python train_crnn.py --resume checkpoints_crnn/checkpoint_epoch_50.pth
```

### CRNN训练特点
- **输入**: 32px高度的文本图像
- **输出**: 文本序列
- **损失函数**: CTC Loss
- **评估指标**: 字符级准确率
- **训练时间**: CPU上约1-2小时（小数据集）

## 🔍 步骤3: 训练CRAFT模型（文本检测）

### 基础训练
```bash
# 使用默认参数训练
python train_craft.py

# 指定参数训练
python train_craft.py \
    --data_dir sample_training_data/craft_data \
    --batch_size 8 \
    --epochs 100 \
    --lr 0.001 \
    --save_dir checkpoints_craft \
    --pretrained
```

### 训练参数说明
- `--data_dir`: 训练数据目录
- `--batch_size`: 批大小（推荐4-8，显存要求高）
- `--epochs`: 训练轮数
- `--lr`: 学习率
- `--save_dir`: 模型保存目录
- `--pretrained`: 使用预训练权重
- `--resume`: 恢复训练

### CRAFT训练特点
- **输入**: 任意尺寸的场景图像
- **输出**: 字符级和链接级热图
- **损失函数**: MSE Loss
- **评估指标**: 检测准确率
- **训练时间**: CPU上约2-4小时（小数据集）

## 📊 训练监控

### 查看训练日志
```python
# CRNN训练示例输出
Epoch 1/100: 100%|██████████| 3/3 [00:02<00:00, 1.20it/s, loss=4.32]
INFO:__main__:Epoch 1, Average Loss: 4.3156
INFO:__main__:Validation Loss: 4.1234, Accuracy: 12.50%
✅ 保存最佳模型，验证损失: 4.1234, 准确率: 12.50%

# CRAFT训练示例输出  
Epoch 1/100: 100%|██████████| 2/2 [00:03<00:00, 1.67s/it, loss=0.45, cls=0.23, geo=0.22]
INFO:__main__:Epoch 1, Loss: 0.4567, Cls: 0.2234, Geo: 0.2333
✅ 保存最佳模型，验证损失: 0.4123
```

### 监控指标
- **CRNN**: Loss下降 + 准确率上升
- **CRAFT**: Loss下降 + 检测框质量提升

## 🔧 步骤4: 使用训练好的模型

### 更新main.py中的模型路径
```python
# 在_init_crnn方法中
model_path = "checkpoints_crnn/best_model.pth"
if model_path and os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=self.device)
    self.crnn_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"加载训练好的CRNN模型: {model_path}")
```

### 测试训练好的模型
```bash
# 使用CRNN方法测试
python main.py --image id.jpeg --method crnn
```

## 📈 训练技巧和优化

### 1. 数据增强
```python
# 在prepare_training_data.py中添加数据增强
from albumentations import *

transform = Compose([
    RandomBrightnessContrast(p=0.5),
    GaussNoise(p=0.3),
    Blur(blur_limit=3, p=0.3),
    Affine(rotate=(-5, 5), p=0.5)
])
```

### 2. 学习率调度
```python
# 在训练脚本中使用更好的学习率调度
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
```

### 3. 早停机制
```python
# 添加早停防止过拟合
class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
```

## 🐛 常见问题解决

### 1. 显存不足
```bash
# 减小批大小
python train_crnn.py --batch_size 8
python train_craft.py --batch_size 4
```

### 2. 训练过慢
```bash
# 减少数据和epochs进行测试
python train_crnn.py --epochs 10
```

### 3. 损失不下降
- 检查学习率是否过大或过小
- 确认数据标注是否正确
- 尝试不同的优化器

### 4. 准确率低
- 增加训练数据量
- 使用数据增强
- 调整模型架构

## 📊 性能基准

### 数据集规模建议
- **小型测试**: 100-500个样本
- **基础应用**: 1000-5000个样本  
- **生产级别**: 10000+个样本

### 训练时间估算
| 数据量 | CRNN (CPU) | CRAFT (CPU) | GPU加速 |
|--------|------------|-------------|---------|
| 100样本 | 30分钟 | 1小时 | 5-10倍 |
| 1000样本 | 2小时 | 4小时 | 5-10倍 |
| 10000样本 | 10小时+ | 20小时+ | 5-10倍 |

## 🎯 完整训练示例

```bash
#!/bin/bash
# 完整训练流程脚本

echo "🔧 准备训练数据..."
python prepare_training_data.py --model both --output my_training_data

echo "🎯 训练CRNN模型..."
python train_crnn.py \
    --data_dir my_training_data/crnn_data/images \
    --label_file my_training_data/crnn_data/labels.txt \
    --batch_size 16 \
    --epochs 50 \
    --save_dir checkpoints_crnn

echo "🔍 训练CRAFT模型..."
python train_craft.py \
    --data_dir my_training_data/craft_data \
    --batch_size 8 \
    --epochs 50 \
    --save_dir checkpoints_craft \
    --pretrained

echo "🧪 测试训练结果..."
python main.py --image id.jpeg --method crnn

echo "🎉 训练完成！"
```

## 📚 进阶指导

### 1. 超参数调优
使用网格搜索或贝叶斯优化找到最佳参数组合。

### 2. 模型集成
训练多个模型并集成结果以提高准确率。

### 3. 迁移学习
使用预训练模型作为起点，在特定数据上微调。

### 4. 模型压缩
使用知识蒸馏或模型剪枝减小模型大小。

---

## 🎉 总结

通过本指南，你可以：
1. ✅ 准备适合的训练数据
2. ✅ 训练CRNN文本识别模型
3. ✅ 训练CRAFT文本检测模型
4. ✅ 集成模型并优化性能

现在开始训练你的身份证识别模型吧！🚀 