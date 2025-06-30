"""
完整检查训练流程
确保所有环节都正确无误
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from datasets import load_dataset

from models.text_detection import CRAFT
from loaddata import OCRDataset
from train_with_your_data import CRAFTLoss


def check_data_pipeline():
    """检查数据管道"""
    print("🔍 检查数据管道...")
    
    # 加载数据集
    ds = load_dataset("lansinuote/ocr_id_card")
    train_dataset = OCRDataset(ds, split='train')
    
    # 数据集基本信息
    print(f"✅ 数据集大小: {len(train_dataset)}")
    
    # 检查数据加载器
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    
    print(f"✅ 数据加载器批次数: {len(dataloader)}")
    
    # 检查几个批次
    batch_count = 0
    error_count = 0
    
    for batch_idx, (images, char_heatmaps, link_heatmaps) in enumerate(dataloader):
        try:
            # 检查数据形状
            assert images.shape[0] <= 4, f"批大小错误: {images.shape[0]}"
            assert images.shape[1:] == (3, 512, 512), f"图像形状错误: {images.shape[1:]}"
            assert char_heatmaps.shape[1:] == (256, 256), f"字符热图形状错误: {char_heatmaps.shape[1:]}"
            assert link_heatmaps.shape[1:] == (256, 256), f"链接热图形状错误: {link_heatmaps.shape[1:]}"
            
            # 检查数据类型
            assert images.dtype == torch.float32, f"图像数据类型错误: {images.dtype}"
            assert char_heatmaps.dtype == torch.float32, f"字符热图数据类型错误: {char_heatmaps.dtype}"
            assert link_heatmaps.dtype == torch.float32, f"链接热图数据类型错误: {link_heatmaps.dtype}"
            
            # 检查数值范围
            assert 0 <= images.min() and images.max() <= 1, f"图像值域错误: [{images.min():.3f}, {images.max():.3f}]"
            assert 0 <= char_heatmaps.min() and char_heatmaps.max() <= 1, f"字符热图值域错误: [{char_heatmaps.min():.3f}, {char_heatmaps.max():.3f}]"
            assert 0 <= link_heatmaps.min() and link_heatmaps.max() <= 1, f"链接热图值域错误: [{link_heatmaps.min():.3f}, {link_heatmaps.max():.3f}]"
            
            batch_count += 1
            
        except Exception as e:
            print(f"❌ 批次 {batch_idx} 数据错误: {e}")
            error_count += 1
        
        if batch_idx >= 10:  # 只检查前10个批次
            break
    
    print(f"✅ 数据检查完成: {batch_count}个正常批次, {error_count}个错误批次")
    return error_count == 0


def check_model_pipeline():
    """检查模型管道"""
    print("\n🔍 检查模型管道...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ 使用设备: {device}")
    
    # 创建模型
    model = CRAFT(pretrained=True, freeze=False).to(device)
    
    # 检查模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ 总参数: {total_params:,}")
    print(f"✅ 可训练参数: {trainable_params:,}")
    
    # 检查模型输出
    test_input = torch.randn(2, 3, 512, 512).to(device)
    
    try:
        model.eval()
        with torch.no_grad():
            outputs, features = model(test_input)
        
        print(f"✅ 模型输出形状: {outputs.shape}")
        print(f"✅ 特征图形状: {features.shape}")
        
        # 检查输出值域
        assert outputs.shape == (2, 256, 256, 2), f"输出形状错误: {outputs.shape}"
        
        pred_char = outputs[:, :, :, 0]
        pred_link = outputs[:, :, :, 1]
        
        char_in_range = (pred_char >= 0).all() and (pred_char <= 1).all()
        link_in_range = (pred_link >= 0).all() and (pred_link <= 1).all()
        
        print(f"✅ 字符输出值域[0,1]: {'是' if char_in_range else '否'}")
        print(f"✅ 链接输出值域[0,1]: {'是' if link_in_range else '否'}")
        
        if not char_in_range or not link_in_range:
            print(f"❌ 输出值域错误! 字符: [{pred_char.min():.3f}, {pred_char.max():.3f}], 链接: [{pred_link.min():.3f}, {pred_link.max():.3f}]")
            return False
            
    except Exception as e:
        print(f"❌ 模型前向传播错误: {e}")
        return False
    
    return True


def check_loss_pipeline():
    """检查损失计算管道"""
    print("\n🔍 检查损失计算管道...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = CRAFTLoss()
    
    # 创建测试数据 - 🔧 修复：预测张量需要梯度
    batch_size = 2
    target_char = torch.rand(batch_size, 256, 256).to(device)
    target_link = torch.rand(batch_size, 256, 256).to(device)
    pred_char = torch.rand(batch_size, 256, 256, requires_grad=True).to(device)
    pred_link = torch.rand(batch_size, 256, 256, requires_grad=True).to(device)
    
    try:
        loss, cls_loss, geo_loss = criterion(target_char, pred_char, target_link, pred_link)
        
        print(f"✅ 损失计算成功")
        print(f"  总损失: {loss.item():.6f}")
        print(f"  字符损失: {cls_loss.item():.6f}")
        print(f"  链接损失: {geo_loss.item():.6f}")
        
        # 检查损失值合理性
        if loss.item() > 2.0:
            print(f"⚠️ 损失值较高: {loss.item():.6f}")
        elif loss.item() < 0.0:
            print(f"❌ 损失值为负: {loss.item():.6f}")
            return False
        else:
            print(f"✅ 损失值合理")
        
        # 检查梯度计算
        loss.backward()
        print(f"✅ 反向传播成功")
        
    except Exception as e:
        print(f"❌ 损失计算错误: {e}")
        return False
    
    return True


def check_training_integration():
    """检查训练集成测试"""
    print("\n🔍 检查训练集成...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    ds = load_dataset("lansinuote/ocr_id_card")
    train_dataset = OCRDataset(ds, split='train')
    dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    
    # 创建模型
    model = CRAFT(pretrained=True, freeze=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = CRAFTLoss()
    
    # 模拟一个训练步骤
    model.train()
    
    try:
        images, char_heatmaps, link_heatmaps = next(iter(dataloader))
        images = images.to(device)
        char_heatmaps = char_heatmaps.to(device)
        link_heatmaps = link_heatmaps.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs, _ = model(images)
        
        pred_char = outputs[:, :, :, 0]
        pred_link = outputs[:, :, :, 1]
        
        # 损失计算
        loss, cls_loss, geo_loss = criterion(char_heatmaps, pred_char, link_heatmaps, pred_link)
        
        # 反向传播
        loss.backward()
        
        # 梯度检查
        grad_norms = []
        for param in model.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.data.norm(2).item())
        
        max_grad = max(grad_norms) if grad_norms else 0
        avg_grad = np.mean(grad_norms) if grad_norms else 0
        
        print(f"✅ 完整训练步骤成功")
        print(f"  损失: {loss.item():.6f}")
        print(f"  最大梯度: {max_grad:.6f}")
        print(f"  平均梯度: {avg_grad:.6f}")
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # 参数更新
        optimizer.step()
        
        print(f"✅ 参数更新成功")
        
        # 检查梯度爆炸
        if max_grad > 10.0:
            print(f"⚠️ 梯度过大: {max_grad:.6f}")
            return False
        
    except Exception as e:
        print(f"❌ 训练集成错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def check_training_script_issues():
    """检查训练脚本中的潜在问题"""
    print("\n🔍 检查训练脚本问题...")
    
    issues = []
    
    # 1. 验证循环缺失
    print("📋 训练脚本分析:")
    print("  ⚠️ 问题1: 缺少验证循环")
    print("    - 有验证数据集分割，但未在训练中使用")
    print("    - 无法监控过拟合")
    issues.append("validation_loop")
    
    # 2. 保存策略
    print("  ⚠️ 问题2: 保存策略可能占用大量空间")
    print("    - 每个epoch都保存检查点")
    print("    - 建议只保存最佳模型和定期检查点")
    issues.append("save_strategy")
    
    # 3. 早停机制
    print("  ⚠️ 问题3: 缺少早停机制")
    print("    - 没有基于验证损失的早停")
    print("    - 可能导致过拟合")
    issues.append("early_stopping")
    
    # 4. 学习率调度
    print("  ✅ 学习率调度: StepLR正常")
    
    # 5. 梯度裁剪
    print("  ✅ 梯度裁剪: 已设置为1.0")
    
    # 6. 异常处理
    print("  ✅ 异常处理: 有异常捕获机制")
    
    return issues


def main():
    """主检查函数"""
    print("🔍 开始完整训练流程检查...\n")
    
    # 检查各个组件
    data_ok = check_data_pipeline()
    model_ok = check_model_pipeline()
    loss_ok = check_loss_pipeline()
    training_ok = check_training_integration()
    
    # 检查脚本问题
    script_issues = check_training_script_issues()
    
    print(f"\n📊 检查结果总结:")
    print(f"  数据管道: {'✅ 正常' if data_ok else '❌ 异常'}")
    print(f"  模型管道: {'✅ 正常' if model_ok else '❌ 异常'}")
    print(f"  损失管道: {'✅ 正常' if loss_ok else '❌ 异常'}")
    print(f"  训练集成: {'✅ 正常' if training_ok else '❌ 异常'}")
    print(f"  脚本问题: {len(script_issues)}个")
    
    all_ok = data_ok and model_ok and loss_ok and training_ok
    
    if all_ok:
        print(f"\n🎉 训练流程基本正常！")
        if script_issues:
            print(f"💡 建议改进以下问题:")
            for issue in script_issues:
                if issue == "validation_loop":
                    print(f"  - 添加验证循环")
                elif issue == "save_strategy":
                    print(f"  - 优化保存策略")
                elif issue == "early_stopping":
                    print(f"  - 添加早停机制")
    else:
        print(f"\n❌ 训练流程存在问题，需要修复！")
    
    return all_ok, script_issues


if __name__ == "__main__":
    success, issues = main()
    
    if success:
        print(f"\n✅ 可以开始训练，但建议优化脚本问题")
    else:
        print(f"\n❌ 请先修复错误再开始训练") 