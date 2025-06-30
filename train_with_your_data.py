"""
使用自定义数据格式训练CRAFT模型
适配box、cls、word格式的数据
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import argparse

from models.text_detection import CRAFT
from loaddata import OCRDataset
from datasets import load_dataset


class CRAFTLoss(nn.Module):
    """CRAFT损失函数"""
    
    def __init__(self):
        super(CRAFTLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, y_true_cls, y_pred_cls, y_true_geo, y_pred_geo):
        # 字符检测损失
        cls_loss = self.mse_loss(y_pred_cls, y_true_cls)
        
        # 链接检测损失
        geo_loss = self.mse_loss(y_pred_geo, y_true_geo)
        
        # 总损失
        total_loss = cls_loss + geo_loss
        
        return total_loss, cls_loss, geo_loss


def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_geo_loss = 0
    
    pbar = tqdm(dataloader, desc='训练中')
    
    for batch_idx, (images, char_heatmaps, link_heatmaps) in enumerate(pbar):
        images = images.to(device)
        char_heatmaps = char_heatmaps.to(device)
        link_heatmaps = link_heatmaps.to(device)
        
        optimizer.zero_grad()
        
        try:
            # 前向传播
            outputs, _ = model(images)
            
            # 提取预测结果
            pred_char = outputs[:, :, :, 0]  # 字符热图
            pred_link = outputs[:, :, :, 1]  # 链接热图
            
            # 计算损失
            loss, cls_loss, geo_loss = criterion(
                char_heatmaps, pred_char,
                link_heatmaps, pred_link
            )
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_geo_loss += geo_loss.item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cls': f'{cls_loss.item():.4f}',
                'geo': f'{geo_loss.item():.4f}'
            })
            
        except Exception as e:
            print(f"训练批次出错: {e}")
            continue
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    avg_cls_loss = total_cls_loss / len(dataloader) if len(dataloader) > 0 else 0
    avg_geo_loss = total_geo_loss / len(dataloader) if len(dataloader) > 0 else 0
    
    return avg_loss, avg_cls_loss, avg_geo_loss


def main():
    parser = argparse.ArgumentParser(description='训练CRAFT模型（自定义数据格式）')
    parser.add_argument('--batch_size', type=int, default=16, help='批大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--save_dir', type=str, default='checkpoints_custom', help='模型保存目录')
    parser.add_argument('--resume', type=str, help='恢复训练的模型路径')
    
    args = parser.parse_args()
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据集
    print("正在加载数据集...")
    ds = load_dataset("lansinuote/ocr_id_card")
    
    # 创建数据集
    train_dataset = OCRDataset(ds, split='train')
    
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
    model = CRAFT(pretrained=True, freeze=False).to(device)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = CRAFTLoss()
    
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
        train_loss, train_cls_loss, train_geo_loss = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        print(f"训练损失: {train_loss:.4f} (字符: {train_cls_loss:.4f}, 链接: {train_geo_loss:.4f})")
        
        # 更新学习率
        scheduler.step()
        
        # 保存检查点
        if train_loss < best_loss:
            best_loss = train_loss
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_cls_loss': train_cls_loss,
                'train_geo_loss': train_geo_loss
            }
            
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"✅ 保存最佳模型，损失: {best_loss:.4f}")
        
        # 定期保存
        if epoch % 1 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_cls_loss': train_cls_loss,
                'train_geo_loss': train_geo_loss
            }
            
            torch.save(checkpoint, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth'))
            print(f"💾 保存检查点: epoch {epoch}")
    
    print("\n🎉 训练完成！")


if __name__ == "__main__":
    main() 