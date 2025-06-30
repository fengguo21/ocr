"""
批次检查点管理工具
用于查看、加载和清理按batch保存的模型检查点
"""

import os
import torch
import argparse
from datetime import datetime
import glob
from models.text_detection import CRAFT


def list_checkpoints(save_dir):
    """列出所有检查点文件"""
    print(f"🔍 扫描目录: {save_dir}")
    
    # 查找所有检查点文件
    batch_checkpoints = glob.glob(os.path.join(save_dir, "checkpoint_epoch_*_batch_*.pth"))
    epoch_checkpoints = glob.glob(os.path.join(save_dir, "best_model.pth"))
    
    print(f"\n📋 检查点文件列表:")
    
    if epoch_checkpoints:
        print(f"\n🏆 最佳模型:")
        for cp in epoch_checkpoints:
            file_size = os.path.getsize(cp) / (1024 * 1024)  # MB
            mod_time = datetime.fromtimestamp(os.path.getmtime(cp))
            print(f"  {os.path.basename(cp)} - {file_size:.1f}MB - {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if batch_checkpoints:
        print(f"\n🔄 批次检查点 ({len(batch_checkpoints)}个):")
        
        # 按时间排序
        batch_checkpoints.sort(key=os.path.getmtime, reverse=True)
        
        total_size = 0
        for cp in batch_checkpoints[:10]:  # 只显示最新的10个
            file_size = os.path.getsize(cp) / (1024 * 1024)  # MB
            total_size += file_size
            mod_time = datetime.fromtimestamp(os.path.getmtime(cp))
            print(f"  {os.path.basename(cp)} - {file_size:.1f}MB - {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if len(batch_checkpoints) > 10:
            print(f"  ... 还有{len(batch_checkpoints) - 10}个文件")
        
        print(f"\n📊 统计信息:")
        print(f"  批次检查点总数: {len(batch_checkpoints)}")
        print(f"  总大小: {sum(os.path.getsize(cp) for cp in batch_checkpoints) / (1024 * 1024):.1f}MB")
    
    return batch_checkpoints, epoch_checkpoints


def load_checkpoint_info(checkpoint_path):
    """加载并显示检查点信息"""
    print(f"\n🔍 检查点详情: {os.path.basename(checkpoint_path)}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Batch: {checkpoint.get('batch', 'N/A')}")
        print(f"  总批次数: {checkpoint.get('total_batches', 'N/A')}")
        print(f"  训练损失: {checkpoint.get('train_loss', 'N/A'):.4f}")
        print(f"  字符损失: {checkpoint.get('cls_loss', 'N/A'):.4f}")
        print(f"  链接损失: {checkpoint.get('geo_loss', 'N/A'):.4f}")
        
        if 'val_loss' in checkpoint:
            print(f"  验证损失: {checkpoint['val_loss']:.4f}")
        
        # 检查模型权重
        state_dict = checkpoint.get('model_state_dict', {})
        print(f"  模型参数数量: {len(state_dict)}")
        
        return checkpoint
        
    except Exception as e:
        print(f"❌ 加载检查点失败: {e}")
        return None


def load_model_from_checkpoint(checkpoint_path, device='cpu'):
    """从检查点加载模型"""
    print(f"🤖 从检查点加载模型: {os.path.basename(checkpoint_path)}")
    
    try:
        # 创建模型
        model = CRAFT(pretrained=False, freeze=False).to(device)
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✅ 模型加载成功！")
        print(f"  - 来自Epoch {checkpoint.get('epoch', 'N/A')}, Batch {checkpoint.get('batch', 'N/A')}")
        print(f"  - 训练损失: {checkpoint.get('train_loss', 'N/A'):.4f}")
        
        return model, checkpoint
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None


def clean_old_checkpoints(save_dir, keep_count=10):
    """清理旧的批次检查点，只保留最新的几个"""
    print(f"🧹 清理旧检查点，保留最新{keep_count}个...")
    
    batch_checkpoints = glob.glob(os.path.join(save_dir, "checkpoint_epoch_*_batch_*.pth"))
    
    if len(batch_checkpoints) <= keep_count:
        print(f"  当前只有{len(batch_checkpoints)}个检查点，无需清理")
        return
    
    # 按时间排序，保留最新的
    batch_checkpoints.sort(key=os.path.getmtime, reverse=True)
    to_delete = batch_checkpoints[keep_count:]
    
    total_size_deleted = 0
    for cp in to_delete:
        try:
            size = os.path.getsize(cp)
            os.remove(cp)
            total_size_deleted += size
            print(f"  ✅ 删除: {os.path.basename(cp)}")
        except Exception as e:
            print(f"  ❌ 删除失败: {os.path.basename(cp)} - {e}")
    
    print(f"🎉 清理完成！删除{len(to_delete)}个文件，释放{total_size_deleted / (1024 * 1024):.1f}MB空间")


def main():
    parser = argparse.ArgumentParser(description='批次检查点管理工具')
    parser.add_argument('--save_dir', type=str, default='checkpoints_improved', help='检查点目录')
    parser.add_argument('--action', choices=['list', 'info', 'load', 'clean'], default='list', help='操作类型')
    parser.add_argument('--checkpoint', type=str, help='特定检查点文件名')
    parser.add_argument('--keep_count', type=int, default=10, help='清理时保留的检查点数量')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        print(f"❌ 目录不存在: {args.save_dir}")
        return
    
    if args.action == 'list':
        list_checkpoints(args.save_dir)
        
    elif args.action == 'info':
        if not args.checkpoint:
            print("❌ 请使用 --checkpoint 指定检查点文件")
            return
        
        checkpoint_path = os.path.join(args.save_dir, args.checkpoint)
        if not os.path.exists(checkpoint_path):
            print(f"❌ 检查点文件不存在: {checkpoint_path}")
            return
        
        load_checkpoint_info(checkpoint_path)
        
    elif args.action == 'load':
        if not args.checkpoint:
            print("❌ 请使用 --checkpoint 指定检查点文件")
            return
        
        checkpoint_path = os.path.join(args.save_dir, args.checkpoint)
        if not os.path.exists(checkpoint_path):
            print(f"❌ 检查点文件不存在: {checkpoint_path}")
            return
        
        model, checkpoint = load_model_from_checkpoint(checkpoint_path)
        if model:
            print(f"💡 模型已加载，可以用于推理或继续训练")
        
    elif args.action == 'clean':
        clean_old_checkpoints(args.save_dir, args.keep_count)
    
    print()


if __name__ == "__main__":
    main() 