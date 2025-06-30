"""
CRAFT模型训练脚本
用于训练身份证文本检测模型
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
import math

from models.text_detection import CRAFT
from utils.image_processing import IDCardPreprocessor


class CRAFTDataset(Dataset):
    """CRAFT文本检测数据集"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # 查找所有图像和对应的标注文件
        self.samples = []
        for img_file in self.data_dir.glob("img_*.jpg"):
            gt_file = self.data_dir / f"gt_{img_file.stem}.txt"
            if gt_file.exists():
                self.samples.append({
                    'image': img_file,
                    'gt': gt_file
                })
        
        print(f"加载 {len(self.samples)} 个训练样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图像
        image = cv2.imread(str(sample['image']))
        if image is None:
            raise ValueError(f"无法加载图像: {sample['image']}")
        
        # 加载标注
        text_boxes = self._load_annotations(sample['gt'])
        
        # 生成训练目标
        char_heatmap, link_heatmap = self._generate_heatmaps(image, text_boxes)
        
        # 图像预处理
        if self.transform:
            image = self.transform(image)
        
        # 调整图像尺寸
        image, char_heatmap, link_heatmap = self._resize_data(
            image, char_heatmap, link_heatmap
        )
        
        # 转换为tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        char_heatmap = torch.from_numpy(char_heatmap).float()
        link_heatmap = torch.from_numpy(link_heatmap).float()
        
        return image, char_heatmap, link_heatmap
    
    def _load_annotations(self, gt_file):
        """加载标注文件"""
        text_boxes = []
        
        with open(gt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(',')
                    if len(parts) >= 8:
                        # 提取坐标点
                        coords = []
                        for i in range(0, 8, 2):
                            x = int(parts[i])
                            y = int(parts[i + 1])
                            coords.append([x, y])
                        
                        # 提取文本（如果有）
                        text = parts[8] if len(parts) > 8 else ""
                        
                        text_boxes.append({
                            'coords': np.array(coords),
                            'text': text
                        })
        
        return text_boxes
    
    def _generate_heatmaps(self, image, text_boxes):
        """生成字符级和链接级热图"""
        h, w = image.shape[:2]
        
        # 初始化热图
        char_heatmap = np.zeros((h, w), dtype=np.float32)
        link_heatmap = np.zeros((h, w), dtype=np.float32)
        
        for box in text_boxes:
            coords = box['coords']
            
            # 生成字符级热图
            self._generate_char_heatmap(char_heatmap, coords)
            
            # 生成链接级热图
            self._generate_link_heatmap(link_heatmap, coords)
        
        return char_heatmap, link_heatmap
    
    def _generate_char_heatmap(self, heatmap, coords):
        """生成字符级热图"""
        # 使用高斯核生成热图
        # 这是一个简化版本，实际CRAFT使用更复杂的方法
        
        # 计算文本框的中心和方向
        center_x = np.mean(coords[:, 0])
        center_y = np.mean(coords[:, 1])
        
        # 计算文本框的宽度和高度
        width = np.linalg.norm(coords[1] - coords[0])
        height = np.linalg.norm(coords[3] - coords[0])
        
        # 生成高斯热图
        sigma = min(width, height) / 6
        self._add_gaussian_heatmap(heatmap, center_x, center_y, sigma)
    
    def _generate_link_heatmap(self, heatmap, coords):
        """生成链接级热图"""
        # 在字符之间生成连接
        # 这里使用简化的方法
        
        # 计算文本框的四个边的中点
        for i in range(len(coords)):
            j = (i + 1) % len(coords)
            
            start_x, start_y = coords[i]
            end_x, end_y = coords[j]
            
            # 在边上生成连接
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2
            
            sigma = min(abs(end_x - start_x), abs(end_y - start_y)) / 4
            if sigma > 0:
                self._add_gaussian_heatmap(heatmap, mid_x, mid_y, sigma)
    
    def _add_gaussian_heatmap(self, heatmap, center_x, center_y, sigma):
        """添加高斯热图"""
        h, w = heatmap.shape
        
        # 计算高斯核的范围
        radius = int(sigma * 3)
        
        for y in range(max(0, int(center_y - radius)), 
                      min(h, int(center_y + radius + 1))):
            for x in range(max(0, int(center_x - radius)), 
                          min(w, int(center_x + radius + 1))):
                
                # 计算高斯值
                dist_sq = (x - center_x)**2 + (y - center_y)**2
                value = math.exp(-dist_sq / (2 * sigma**2))
                
                # 取最大值
                heatmap[y, x] = max(heatmap[y, x], value)
    
    def _resize_data(self, image, char_heatmap, link_heatmap, input_size=512, target_size=256):
        """
        调整数据尺寸
        input_size: 输入图像尺寸 (用于VGG16特征提取)
        target_size: 目标热图尺寸 (匹配CRAFT输出)
        """
        h, w = image.shape[:2]
        
        # 计算缩放比例
        scale = input_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # 调整图像尺寸到input_size
        image = cv2.resize(image, (new_w, new_h))
        
        # 填充图像到input_size
        pad_h = input_size - new_h
        pad_w = input_size - new_w
        
        image = cv2.copyMakeBorder(
            image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
        )
        
        # 调整热图尺寸到target_size (匹配CRAFT输出)
        target_scale = target_size / max(h, w)
        target_new_h, target_new_w = int(h * target_scale), int(w * target_scale)
        
        char_heatmap = cv2.resize(char_heatmap, (target_new_w, target_new_h))
        link_heatmap = cv2.resize(link_heatmap, (target_new_w, target_new_h))
        
        # 填充热图到target_size
        target_pad_h = target_size - target_new_h
        target_pad_w = target_size - target_new_w
        
        char_heatmap = cv2.copyMakeBorder(
            char_heatmap, 0, target_pad_h, 0, target_pad_w, cv2.BORDER_CONSTANT, value=0
        )
        link_heatmap = cv2.copyMakeBorder(
            link_heatmap, 0, target_pad_h, 0, target_pad_w, cv2.BORDER_CONSTANT, value=0
        )
        
        return image, char_heatmap, link_heatmap


class CRAFTLoss(nn.Module):
    """CRAFT损失函数"""
    
    def __init__(self):
        super(CRAFTLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, y_true_cls, y_pred_cls, y_true_geo, y_pred_geo):
        """
        计算CRAFT损失
        
        Args:
            y_true_cls: 真实字符热图
            y_pred_cls: 预测字符热图  
            y_true_geo: 真实链接热图
            y_pred_geo: 预测链接热图
        """
        # 字符检测损失
        cls_loss = self.mse_loss(y_pred_cls, y_true_cls)
        
        # 链接检测损失
        geo_loss = self.mse_loss(y_pred_geo, y_true_geo)
        
        # 总损失
        total_loss = cls_loss + geo_loss
        
        return total_loss, cls_loss, geo_loss


class CRAFTTrainer:
    """CRAFT训练器"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.criterion = CRAFTLoss()
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, dataloader, optimizer, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_cls_loss = 0
        total_geo_loss = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, (images, char_heatmaps, link_heatmaps) in enumerate(pbar):
            images = images.to(self.device)
            char_heatmaps = char_heatmaps.to(self.device)
            link_heatmaps = link_heatmaps.to(self.device)
            
            optimizer.zero_grad()
            
            try:
                # 前向传播
                outputs, _ = self.model(images)
                
                # 提取预测结果 (outputs shape: [batch, height, width, 2])
                pred_char = outputs[:, :, :, 0]  # 字符热图 [batch, height, width]
                pred_link = outputs[:, :, :, 1]  # 链接热图 [batch, height, width]
                
                # 计算损失 (target和prediction都是[batch, height, width])
                loss, cls_loss, geo_loss = self.criterion(
                    char_heatmaps, pred_char,
                    link_heatmaps, pred_link
                )
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                total_cls_loss += cls_loss.item()
                total_geo_loss += geo_loss.item()
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': loss.item(),
                    'cls': cls_loss.item(),
                    'geo': geo_loss.item()
                })
                
            except Exception as e:
                print(f"训练批次出错: {e}")
                continue
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        avg_cls_loss = total_cls_loss / len(dataloader) if len(dataloader) > 0 else 0
        avg_geo_loss = total_geo_loss / len(dataloader) if len(dataloader) > 0 else 0
        
        self.logger.info(
            f'Epoch {epoch}, Loss: {avg_loss:.4f}, '
            f'Cls: {avg_cls_loss:.4f}, Geo: {avg_geo_loss:.4f}'
        )
        
        return avg_loss, avg_cls_loss, avg_geo_loss
    
    def validate(self, dataloader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        total_cls_loss = 0
        total_geo_loss = 0
        
        with torch.no_grad():
            for images, char_heatmaps, link_heatmaps in dataloader:
                images = images.to(self.device)
                char_heatmaps = char_heatmaps.to(self.device)
                link_heatmaps = link_heatmaps.to(self.device)
                
                try:
                    # 前向传播
                    outputs, _ = self.model(images)
                    
                    # 提取预测结果 (outputs shape: [batch, height, width, 2])
                    pred_char = outputs[:, :, :, 0]  # 字符热图 [batch, height, width]
                    pred_link = outputs[:, :, :, 1]  # 链接热图 [batch, height, width]
                    
                    # 计算损失 (target和prediction都是[batch, height, width])
                    loss, cls_loss, geo_loss = self.criterion(
                        char_heatmaps, pred_char,
                        link_heatmaps, pred_link
                    )
                    
                    total_loss += loss.item()
                    total_cls_loss += cls_loss.item()
                    total_geo_loss += geo_loss.item()
                    
                except Exception as e:
                    print(f"验证批次出错: {e}")
                    continue
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        avg_cls_loss = total_cls_loss / len(dataloader) if len(dataloader) > 0 else 0
        avg_geo_loss = total_geo_loss / len(dataloader) if len(dataloader) > 0 else 0
        
        self.logger.info(
            f'Validation Loss: {avg_loss:.4f}, '
            f'Cls: {avg_cls_loss:.4f}, Geo: {avg_geo_loss:.4f}'
        )
        
        return avg_loss, avg_cls_loss, avg_geo_loss


def main():
    parser = argparse.ArgumentParser(description='训练CRAFT模型')
    parser.add_argument('--data_dir', type=str, help='训练数据目录')
    parser.add_argument('--batch_size', type=int, default=8, help='批大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--save_dir', type=str, default='checkpoints_craft', help='模型保存目录')
    parser.add_argument('--resume', type=str, help='恢复训练的模型路径')
    parser.add_argument('--pretrained', action='store_true', help='使用预训练权重')
    
    args = parser.parse_args()
    
    # 默认使用生成的训练数据
    if not args.data_dir:
        args.data_dir = 'sample_training_data/craft_data'
    
    # 检查数据目录是否存在
    if not os.path.exists(args.data_dir):
        print(f"错误: 数据目录不存在: {args.data_dir}")
        print("请先运行: python prepare_training_data.py --model craft")
        return
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据集
    dataset = CRAFTDataset(args.data_dir)
    
    if len(dataset) == 0:
        print("错误: 没有找到训练数据")
        return
    
    # 分割训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")
    
    # 创建模型
    model = CRAFT(pretrained=args.pretrained, freeze=False).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # 训练器
    trainer = CRAFTTrainer(model, device)
    
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
        train_loss, train_cls_loss, train_geo_loss = trainer.train_epoch(
            train_loader, optimizer, epoch
        )
        
        # 验证
        val_loss, val_cls_loss, val_geo_loss = trainer.validate(val_loader)
        
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
                'train_cls_loss': train_cls_loss,
                'train_geo_loss': train_geo_loss,
                'val_cls_loss': val_cls_loss,
                'val_geo_loss': val_geo_loss
            }
            
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"✅ 保存最佳模型，验证损失: {best_loss:.4f}")
        
        # 定期保存
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_cls_loss': train_cls_loss,
                'train_geo_loss': train_geo_loss,
                'val_cls_loss': val_cls_loss,
                'val_geo_loss': val_geo_loss
            }
            
            torch.save(checkpoint, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth'))
            print(f"💾 保存检查点: epoch_{epoch}.pth")
    
    print("\n🎉 训练完成！")
    print(f"最佳模型保存在: {args.save_dir}/best_model.pth")


if __name__ == "__main__":
    main() 