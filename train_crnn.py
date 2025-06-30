"""
CRNN模型训练脚本
用于训练身份证文本识别模型
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
import json
import argparse
from tqdm import tqdm
import logging
from pathlib import Path

from models.crnn import CRNN
from utils.image_processing import IDCardPreprocessor


class TextDataset(Dataset):
    """文本识别数据集"""
    
    def __init__(self, data_dir, label_file, charset, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.charset = charset
        self.char_to_idx = {char: idx for idx, char in enumerate(charset)}
        
        # 加载标签 - 支持txt格式
        self.samples = []
        if label_file.endswith('.txt'):
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split(' ', 1)
                        if len(parts) == 2:
                            image_name, text = parts
                            self.samples.append({
                                'image': image_name,
                                'text': text
                            })
        else:
            # JSON格式
            with open(label_file, 'r', encoding='utf-8') as f:
                self.samples = json.load(f)
        
        print(f"加载 {len(self.samples)} 个训练样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = self.data_dir / sample['image']
        text = sample['text']
        
        # 加载图像
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        
        # 图像预处理
        if self.transform:
            image = self.transform(image)
        
        # 调整图像尺寸
        image = self._resize_image(image)
        
        # 转换为tensor
        image = torch.from_numpy(image).float() / 255.0
        image = image.unsqueeze(0)  # 添加通道维度
        
        # 编码文本
        encoded_text = self._encode_text(text)
        
        return image, encoded_text, text
    
    def _resize_image(self, image, target_height=32):
        """调整图像尺寸"""
        h, w = image.shape
        target_width = int(w * target_height / h)
        
        # 确保宽度在合理范围内
        min_width = 16
        max_width = 160
        target_width = max(min_width, min(target_width, max_width))
        
        resized = cv2.resize(image, (target_width, target_height))
        return resized
    
    def _encode_text(self, text):
        """编码文本为数字序列"""
        encoded = []
        for char in text:
            if char in self.char_to_idx:
                encoded.append(self.char_to_idx[char])
            else:
                # 未知字符用0表示
                print(f"警告: 未知字符 '{char}' 在文本 '{text}' 中")
                encoded.append(0)
        return encoded


def collate_fn(batch):
    """自定义批处理函数"""
    images, targets, texts = zip(*batch)
    
    # 获取批次中的最大宽度
    max_width = max(img.shape[2] for img in images)
    batch_size = len(images)
    
    # 创建填充后的图像tensor
    padded_images = torch.zeros(batch_size, 1, 32, max_width)
    target_lengths = []
    all_targets = []
    
    for i, (img, target, text) in enumerate(zip(images, targets, texts)):
        # 填充图像
        padded_images[i, :, :, :img.shape[2]] = img
        
        # 收集目标
        all_targets.extend(target)
        target_lengths.append(len(target))
    
    # 计算输入长度（基于图像宽度）
    input_lengths = [max_width // 4 for _ in images]  # CNN下采样4倍
    
    return (padded_images, 
            torch.tensor(all_targets, dtype=torch.long),
            torch.tensor(input_lengths, dtype=torch.long),
            torch.tensor(target_lengths, dtype=torch.long),
            texts)


class CTCTrainer:
    """CTC训练器"""
    
    def __init__(self, model, device, charset):
        self.model = model
        self.device = device
        self.charset = charset
        self.criterion = nn.CTCLoss(blank=len(charset), zero_infinity=True)
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, dataloader, optimizer, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, (images, targets, input_lengths, target_lengths, texts) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            input_lengths = input_lengths.to(self.device)
            target_lengths = target_lengths.to(self.device)
            
            optimizer.zero_grad()
            
            try:
                # 前向传播
                outputs = self.model(images)
                outputs = outputs.log_softmax(2)
                
                # 计算CTC损失
                loss = self.criterion(outputs, targets, input_lengths, target_lengths)
                
                # 检查损失是否有效
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"跳过无效损失: {loss.item()}")
                    continue
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                
                # 更新进度条
                pbar.set_postfix({'loss': loss.item()})
                
            except Exception as e:
                print(f"训练批次出错: {e}")
                continue
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        self.logger.info(f'Epoch {epoch}, Average Loss: {avg_loss:.4f}')
        
        return avg_loss
    
    def validate(self, dataloader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, targets, input_lengths, target_lengths, texts in dataloader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                input_lengths = input_lengths.to(self.device)
                target_lengths = target_lengths.to(self.device)
                
                try:
                    # 前向传播
                    outputs = self.model(images)
                    outputs = outputs.log_softmax(2)
                    
                    # 计算损失
                    loss = self.criterion(outputs, targets, input_lengths, target_lengths)
                    total_loss += loss.item()
                    
                    # 解码预测结果
                    predicted_texts = self._decode_predictions(outputs)
                    
                    # 计算准确率
                    for pred, gt in zip(predicted_texts, texts):
                        if pred == gt:
                            correct += 1
                        total += 1
                        
                except Exception as e:
                    print(f"验证批次出错: {e}")
                    continue
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        accuracy = correct / total * 100 if total > 0 else 0
        
        self.logger.info(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        return avg_loss, accuracy
    
    def _decode_predictions(self, outputs):
        """解码预测结果"""
        predictions = []
        outputs = outputs.detach().cpu().numpy()
        
        # 转置维度：(seq_len, batch, class) -> (batch, seq_len, class)
        outputs = outputs.transpose(1, 0, 2)
        
        for output in outputs:
            # 简单的贪心解码
            pred_chars = []
            prev_char = None
            
            for t in range(output.shape[0]):
                char_idx = np.argmax(output[t])
                
                # 跳过blank和重复字符
                if char_idx != len(self.charset) and char_idx != prev_char:
                    if char_idx < len(self.charset):
                        pred_chars.append(self.charset[char_idx])
                
                prev_char = char_idx
            
            predictions.append(''.join(pred_chars))
        
        return predictions


def create_sample_dataset():
    """创建示例数据集"""
    # 创建示例数据用于测试
    sample_data = [
        {"image": "sample1.jpg", "text": "张三"},
        {"image": "sample2.jpg", "text": "李四"},
        {"image": "sample3.jpg", "text": "1234567890"},
        {"image": "sample4.jpg", "text": "男"},
        {"image": "sample5.jpg", "text": "汉族"},
    ]
    
    os.makedirs('data/train', exist_ok=True)
    with open('data/train/labels.json', 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    # 也创建txt格式
    with open('data/train/labels.txt', 'w', encoding='utf-8') as f:
        for item in sample_data:
            f.write(f"{item['image']} {item['text']}\n")
    
    print("示例数据集已创建 (JSON和TXT格式)")


def main():
    parser = argparse.ArgumentParser(description='训练CRNN模型')
    parser.add_argument('--data_dir', type=str, help='训练数据目录')
    parser.add_argument('--label_file', type=str, help='标签文件')
    parser.add_argument('--batch_size', type=int, default=16, help='批大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='模型保存目录')
    parser.add_argument('--resume', type=str, help='恢复训练的模型路径')
    parser.add_argument('--create_sample', action='store_true', help='创建示例数据集')
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_dataset()
        return
    
    # 默认使用生成的训练数据
    if not args.data_dir:
        args.data_dir = 'sample_training_data/crnn_data/images'
    if not args.label_file:
        args.label_file = 'sample_training_data/crnn_data/labels.txt'
    
    # 检查数据文件是否存在
    if not os.path.exists(args.data_dir):
        print(f"错误: 数据目录不存在: {args.data_dir}")
        print("请先运行: python prepare_training_data.py --model crnn")
        return
    
    if not os.path.exists(args.label_file):
        print(f"错误: 标签文件不存在: {args.label_file}")
        return
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 字符集
    charset = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' + \
              '中华人民共和国身份证姓名性别民族出生年月日住址公民号码签发机关有效期限长至汉族男女' + \
              '省市县区街道路号楼室派出所公安局厅'
    
    print(f"字符集大小: {len(charset)}")
    
    # 创建数据集
    preprocessor = IDCardPreprocessor()
    
    train_dataset = TextDataset(
        args.data_dir, 
        args.label_file, 
        charset,
        transform=None  # 暂时不使用变换
    )
    
    # 分割训练集和验证集
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # 设为0避免多进程问题
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")
    
    # 创建模型
    model = CRNN(
        img_h=32,
        nc=1,
        nclass=len(charset) + 1,  # +1 for blank
        nh=256
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # 训练器
    trainer = CTCTrainer(model, device, charset)
    
    # 恢复训练
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"从epoch {start_epoch}恢复训练")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 训练循环
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        
        # 训练
        train_loss = trainer.train_epoch(train_loader, optimizer, epoch)
        
        # 验证
        val_loss, val_accuracy = trainer.validate(val_loader)
        
        # 更新学习率
        scheduler.step()
        
        # 保存检查点
        if val_loss < best_loss:
            best_loss = val_loss
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'charset': charset
            }
            
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"✅ 保存最佳模型，验证损失: {best_loss:.4f}, 准确率: {val_accuracy:.2f}%")
        
        # 定期保存
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'charset': charset
            }
            
            torch.save(checkpoint, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth'))
            print(f"💾 保存检查点: epoch_{epoch}.pth")
    
    print("\n🎉 训练完成！")
    print(f"最佳模型保存在: {args.save_dir}/best_model.pth")


if __name__ == "__main__":
    main() 