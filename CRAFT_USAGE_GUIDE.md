# CRAFT模型训练和使用指南

## 🚀 完整流程

### 1. 训练您自己的CRAFT模型

```bash
# 训练模型（使用您的OCR数据）
python train_with_your_data.py --batch_size 4 --epochs 50 --lr 0.001

# 查看训练进度
# 训练过程中会显示进度条和损失值
# 最佳模型会自动保存到 checkpoints_custom/best_model.pth
```

### 2. 使用训练好的模型

#### 方法一：使用简单示例脚本

```bash
# 使用训练好的权重
python use_trained_model.py
```

#### 方法二：使用命令行推理脚本

```bash
# 基础用法
python inference_craft.py --image_path your_image.jpg --model_path checkpoints_custom/best_model.pth

# 自定义参数
python inference_craft.py \
    --image_path id.jpeg \
    --model_path checkpoints_custom/best_model.pth \
    --output_dir results \
    --text_threshold 0.7 \
    --link_threshold 0.4 \
    --device cpu
```

#### 方法三：在代码中使用

```python
from models.text_detection import TextDetector

# 创建检测器（使用训练好的权重）
detector = TextDetector(
    model_path="checkpoints_custom/best_model.pth",
    device='cpu',
    use_pretrained=True
)

# 进行检测
boxes, polys = detector.detect_text(
    image,
    text_threshold=0.7,
    link_threshold=0.4
)
```

## 📁 文件说明

### 训练相关文件
- `train_with_your_data.py` - 主训练脚本
- `loaddata.py` - 数据加载器（适配您的数据格式）
- `models/text_detection.py` - CRAFT模型定义

### 推理相关文件
- `inference_craft.py` - 完整的推理脚本
- `use_trained_model.py` - 简单使用示例
- `models/text_detection.py` - 已修改支持加载自定义权重

### 数据和结果
- `checkpoints_custom/` - 训练生成的模型权重
- `detection_results/` - 推理结果输出目录
- `heatmap_visualization.png` - 热力图可视化

## 🔧 不同使用场景

### 场景1：您已经训练了模型
```python
# 加载您训练的权重
detector = TextDetector(
    model_path="checkpoints_custom/best_model.pth",
    device='cpu'
)
```

### 场景2：使用预训练权重
```python
# 使用ImageNet预训练的VGG16权重
detector = TextDetector(
    model_path=None,
    device='cpu',
    use_pretrained=True
)
```

### 场景3：从头开始（随机权重）
```python
# 完全随机初始化
detector = TextDetector(
    model_path=None,
    device='cpu',
    use_pretrained=False
)
```

## 📊 模型权重格式

您的训练脚本保存的模型文件包含：

```python
checkpoint = {
    'epoch': epoch,                    # 训练轮数
    'model_state_dict': model.state_dict(),  # 模型权重
    'optimizer_state_dict': optimizer.state_dict(),  # 优化器状态
    'train_loss': train_loss,          # 训练损失
    'train_cls_loss': train_cls_loss,  # 字符检测损失
    'train_geo_loss': train_geo_loss   # 链接检测损失
}
```

加载时会自动检测格式并提取模型权重。

## 🎯 参数调优

### 检测阈值调整
- `text_threshold=0.7` - 文本区域检测阈值，越高越严格
- `link_threshold=0.4` - 文本链接检测阈值
- `low_text=0.4` - 低置信度文本阈值

### 训练参数调整
- `--batch_size 4` - 批次大小，根据显存调整
- `--epochs 50` - 训练轮数
- `--lr 0.001` - 学习率

## 🚨 常见问题

### Q: 如何知道模型是否正确加载？
A: 脚本会输出详细信息：
```
正在加载自定义模型权重: checkpoints_custom/best_model.pth
✅ 加载训练好的模型 - Epoch: 49
训练损失: 0.1234
```

### Q: 检测效果不好怎么办？
A: 尝试调整阈值：
```python
boxes, polys = detector.detect_text(
    image,
    text_threshold=0.5,  # 降低阈值
    link_threshold=0.3,
    low_text=0.3
)
```

### Q: 如何在GPU上运行？
A: 设置device参数：
```python
detector = TextDetector(
    model_path="checkpoints_custom/best_model.pth",
    device='cuda'  # 使用GPU
)
```

## 📈 训练建议

1. **数据质量**：确保标注数据准确
2. **训练时间**：建议至少训练50个epoch
3. **学习率**：从0.001开始，根据损失曲线调整
4. **批次大小**：根据显存大小调整（推荐4-8）

## 🎉 完整示例

```python
# 1. 训练模型
# python train_with_your_data.py --epochs 50

# 2. 使用训练好的模型
from models.text_detection import TextDetector

detector = TextDetector("checkpoints_custom/best_model.pth")
boxes, polys = detector.detect_text("your_image.jpg")
print(f"检测到 {len(boxes)} 个文本区域")
```

现在您可以完美使用训练好的CRAFT模型了！🚀 