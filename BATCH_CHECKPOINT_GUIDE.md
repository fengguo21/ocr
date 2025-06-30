# 批次检查点训练指南

## 概述

改进版训练脚本现在支持每100个batch自动保存模型检查点，提供更细粒度的训练进度控制。

## 功能特性

### ✅ 新增功能
- **批次级保存**：每100个batch（可配置）自动保存模型
- **详细信息记录**：保存epoch、batch、损失等详细信息
- **检查点管理**：专用工具管理和清理检查点文件
- **灵活恢复**：可从任意批次检查点恢复训练

### 📋 检查点信息
每个批次检查点包含：
- `epoch`: 当前轮数
- `batch`: 当前批次
- `total_batches`: 总批次数
- `model_state_dict`: 模型权重
- `optimizer_state_dict`: 优化器状态
- `train_loss`: 平均训练损失
- `cls_loss`: 字符检测损失
- `geo_loss`: 链接检测损失

## 使用方法

### 1. 训练模型（默认每100个batch保存）

```bash
# 基础训练，每100个batch保存一次
python train_with_your_data_improved.py --epochs 50 --batch_size 16

# 自定义保存频率，每50个batch保存一次
python train_with_your_data_improved.py --epochs 50 --batch_size 16 --save_freq 50

# 每200个batch保存一次（较少保存）
python train_with_your_data_improved.py --epochs 50 --batch_size 16 --save_freq 200
```

### 2. 查看检查点文件

```bash
# 列出所有检查点
python manage_batch_checkpoints.py --action list

# 查看特定检查点详细信息
python manage_batch_checkpoints.py --action info --checkpoint checkpoint_epoch_1_batch_100.pth

# 指定检查点目录
python manage_batch_checkpoints.py --save_dir checkpoints_improved --action list
```

### 3. 加载检查点

```bash
# 加载特定批次检查点
python manage_batch_checkpoints.py --action load --checkpoint checkpoint_epoch_1_batch_500.pth
```

### 4. 清理旧检查点

```bash
# 保留最新10个检查点，删除其余
python manage_batch_checkpoints.py --action clean --keep_count 10

# 保留最新5个检查点
python manage_batch_checkpoints.py --action clean --keep_count 5
```

## 文件命名规则

检查点文件按以下格式命名：
- 批次检查点：`checkpoint_epoch_{epoch}_batch_{batch}.pth`
- 最佳模型：`best_model.pth`

示例：
- `checkpoint_epoch_1_batch_100.pth` - 第1轮第100个batch
- `checkpoint_epoch_2_batch_300.pth` - 第2轮第300个batch
- `best_model.pth` - 验证损失最低的模型

## 恢复训练

如果训练中断，可以从最新的检查点恢复：

```bash
# 从最佳模型恢复
python train_with_your_data_improved.py --resume checkpoints_improved/best_model.pth

# 从特定批次检查点恢复（需要修改脚本支持）
# 注意：当前版本只支持从epoch级检查点恢复，批次级恢复需要额外实现
```

## 存储空间管理

### 空间占用
- 每个检查点约65MB（CRAFT模型大小）
- 每100个batch = 1个检查点
- 1个epoch（1250个batch）= 12-13个检查点 ≈ 800MB

### 建议策略
1. **训练中**：保持所有批次检查点用于调试
2. **训练后**：只保留关键检查点
   ```bash
   # 只保留最新5个检查点
   python manage_batch_checkpoints.py --action clean --keep_count 5
   ```

## 实际应用场景

### 场景1：长时间训练监控
```bash
# 小批量训练，密切监控
python train_with_your_data_improved.py --batch_size 8 --save_freq 50
```

### 场景2：稳定训练
```bash
# 标准配置
python train_with_your_data_improved.py --batch_size 16 --save_freq 100
```

### 场景3：快速实验
```bash
# 较少保存，节省空间
python train_with_your_data_improved.py --batch_size 32 --save_freq 200
```

## 检查点管理最佳实践

### 1. 定期清理
```bash
# 每训练几个epoch后清理
python manage_batch_checkpoints.py --action clean --keep_count 10
```

### 2. 监控空间
```bash
# 检查检查点总大小
python manage_batch_checkpoints.py --action list
```

### 3. 备份重要检查点
重要的训练阶段检查点应该手动备份到其他位置。

## 注意事项

1. **磁盘空间**：批次检查点会占用大量空间，注意监控
2. **I/O开销**：频繁保存会增加I/O负担，影响训练速度
3. **文件管理**：使用管理工具定期清理旧文件
4. **恢复限制**：当前只支持从epoch级检查点恢复，批次级恢复需要额外开发

## 故障排除

### 问题1：磁盘空间不足
```bash
# 清理旧检查点
python manage_batch_checkpoints.py --action clean --keep_count 5
```

### 问题2：检查点文件损坏
```bash
# 查看检查点信息
python manage_batch_checkpoints.py --action info --checkpoint [文件名]
```

### 问题3：训练速度变慢
- 减少保存频率：`--save_freq 200`
- 检查磁盘I/O性能

## 示例输出

### 训练过程
```
Epoch 1 - 训练 [已保存: batch 100]: 8%|████▌ | 100/1250 [02:30<28:45, 1.50s/it, loss=0.4123, cls=0.2056, geo=0.2067, errors=0, avg_loss=0.4200]
```

### 检查点列表
```
🔍 扫描目录: checkpoints_improved

📋 检查点文件列表:

🏆 最佳模型:
  best_model.pth - 65.2MB - 2024-01-15 14:30:45

🔄 批次检查点 (15个):
  checkpoint_epoch_2_batch_300.pth - 65.2MB - 2024-01-15 14:35:20
  checkpoint_epoch_2_batch_200.pth - 65.2MB - 2024-01-15 14:32:10
  checkpoint_epoch_2_batch_100.pth - 65.2MB - 2024-01-15 14:29:00
  ...

📊 统计信息:
  批次检查点总数: 15
  总大小: 978.0MB
``` 