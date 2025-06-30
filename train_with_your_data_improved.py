"""
改进版CRAFT训练脚本
- 添加验证循环
- 早停机制
- 优化保存策略
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import argparse
import time
from collections import defaultdict

from models.text_detection import CRAFT
from loaddata import OCRDataset
from datasets import load_dataset


class CRAFTLoss(nn.Module):
    """CRAFT损失函数"""
    
    def __init__(self):
        super(CRAFTLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, y_true_cls, y_pred_cls, y_true_geo, y_pred_geo):
        cls_loss = self.mse_loss(y_pred_cls, y_true_cls)
        geo_loss = self.mse_loss(y_pred_geo, y_true_geo)
        total_loss = cls_loss + geo_loss
        return total_loss, cls_loss, geo_loss


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_geo_loss = 0
    batch_count = 0
    error_count = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} - 训练')
    
    for batch_idx, (images, char_heatmaps, link_heatmaps) in enumerate(pbar):
        try:
            images = images.to(device)
            char_heatmaps = char_heatmaps.to(device)
            link_heatmaps = link_heatmaps.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs, _ = model(images)
            
            # 提取预测结果
            pred_char = outputs[:, :, :, 0]
            pred_link = outputs[:, :, :, 1]
            
            # 计算损失
            loss, cls_loss, geo_loss = criterion(
                char_heatmaps, pred_char,
                link_heatmaps, pred_link
            )
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_geo_loss += geo_loss.item()
            batch_count += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cls': f'{cls_loss.item():.4f}',
                'geo': f'{geo_loss.item():.4f}',
                'errors': error_count
            })
            
        except Exception as e:
            error_count += 1
            print(f"训练批次 {batch_idx} 出错: {e}")
            continue
    
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    avg_cls_loss = total_cls_loss / batch_count if batch_count > 0 else 0
    avg_geo_loss = total_geo_loss / batch_count if batch_count > 0 else 0
    
    return avg_loss, avg_cls_loss, avg_geo_loss, error_count


def validate_epoch(model, dataloader, criterion, device, epoch):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    total_cls_loss = 0
    total_geo_loss = 0
    batch_count = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} - 验证')
    
    with torch.no_grad():
        for batch_idx, (images, char_heatmaps, link_heatmaps) in enumerate(pbar):
            try:
                images = images.to(device)
                char_heatmaps = char_heatmaps.to(device)
                link_heatmaps = link_heatmaps.to(device)
                
                outputs, _ = model(images)
                pred_char = outputs[:, :, :, 0]
                pred_link = outputs[:, :, :, 1]
                
                loss, cls_loss, geo_loss = criterion(
                    char_heatmaps, pred_char,
                    link_heatmaps, pred_link
                )
                
                total_loss += loss.item()
                total_cls_loss += cls_loss.item()
                total_geo_loss += geo_loss.item()
                batch_count += 1
                
                pbar.set_postfix({
                    'val_loss': f'{loss.item():.4f}',
                    'val_cls': f'{cls_loss.item():.4f}',
                    'val_geo': f'{geo_loss.item():.4f}'
                })
                
            except Exception as e:
                print(f"验证批次 {batch_idx} 出错: {e}")
                continue
    
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    avg_cls_loss = total_cls_loss / batch_count if batch_count > 0 else 0
    avg_geo_loss = total_geo_loss / batch_count if batch_count > 0 else 0
    
    return avg_loss, avg_cls_loss, avg_geo_loss


def main():
    parser = argparse.ArgumentParser(description='改进版CRAFT训练脚本')
    parser.add_argument('--batch_size', type=int, default=16, help='批大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--save_dir', type=str, default='checkpoints_improved', help='模型保存目录')
    parser.add_argument('--resume', type=str, help='恢复训练的模型路径')
    parser.add_argument('--patience', type=int, default=7, help='早停耐心值')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 使用设备: {device}")
    
    # 加载数据集
    print("📊 正在加载数据集...")
    ds = load_dataset("lansinuote/ocr_id_card")
    full_dataset = OCRDataset(ds, split='train')
    
    # 分割训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"📈 训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")
    
    # 创建模型
    model = CRAFT(pretrained=True, freeze=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = CRAFTLoss()
    early_stopping = EarlyStopping(patience=args.patience)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    
    print(f"\n🚀 开始训练...")
    
    for epoch in range(args.epochs):
        # 训练
        train_loss, train_cls_loss, train_geo_loss, train_errors = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # 验证
        val_loss, val_cls_loss, val_geo_loss = validate_epoch(
            model, val_loader, criterion, device, epoch
        )
        
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"训练 - 总损失: {train_loss:.4f}, 字符: {train_cls_loss:.4f}, 链接: {train_geo_loss:.4f}")
        print(f"验证 - 总损失: {val_loss:.4f}, 字符: {val_cls_loss:.4f}, 链接: {val_geo_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"✅ 保存最佳模型，验证损失: {best_val_loss:.4f}")
        
        # 早停检查
        if early_stopping(val_loss):
            print(f"\n🛑 早停触发！验证损失已连续{args.patience}个epoch未改善")
            break
    
    print(f"\n🎉 训练完成！最佳验证损失: {best_val_loss:.4f}")


if __name__ == "__main__":
    main() 