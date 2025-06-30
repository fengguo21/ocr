"""
可视化热力图生成效果
"""

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from loaddata import OCRDataset
import torch

# 加载数据
ds = load_dataset("lansinuote/ocr_id_card")
dataset = OCRDataset(ds, split='train')

# 获取一个样本
idx = 0
image, char_heatmap, link_heatmap = dataset[idx]

# 转换为numpy格式用于显示
image_np = image.permute(1, 2, 0).numpy()
char_heatmap_np = char_heatmap.numpy()
link_heatmap_np = link_heatmap.numpy()

# 创建可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 显示原图
axes[0].imshow(image_np)
axes[0].set_title('原图')
axes[0].axis('off')

# 显示字符热图
im1 = axes[1].imshow(char_heatmap_np, cmap='hot', interpolation='nearest')
axes[1].set_title(f'字符热图 (max: {char_heatmap_np.max():.3f})')
axes[1].axis('off')
plt.colorbar(im1, ax=axes[1])

# 显示链接热图
im2 = axes[2].imshow(link_heatmap_np, cmap='hot', interpolation='nearest')
axes[2].set_title(f'链接热图 (max: {link_heatmap_np.max():.3f})')
axes[2].axis('off')
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig('heatmap_visualization.png', dpi=150, bbox_inches='tight')
print("✅ 热力图可视化已保存为 'heatmap_visualization.png'")

# 打印统计信息
print(f"\n📊 热力图统计:")
print(f"图像尺寸: {image_np.shape}")
print(f"字符热图尺寸: {char_heatmap_np.shape}")
print(f"字符热图值范围: [{char_heatmap_np.min():.3f}, {char_heatmap_np.max():.3f}]")
print(f"字符热图非零像素数: {np.count_nonzero(char_heatmap_np)}")
print(f"链接热图值范围: [{link_heatmap_np.min():.3f}, {link_heatmap_np.max():.3f}]")
print(f"链接热图非零像素数: {np.count_nonzero(link_heatmap_np)}")

# 检查数据集的标注信息
raw_item = ds['train'][idx]
print(f"\n📝 原始标注信息:")
print(f"OCR结果数量: {len(raw_item['ocr'])}")
for i, ocr in enumerate(raw_item['ocr'][:3]):  # 显示前3个
    print(f"  {i+1}. {ocr['word']} - box: {ocr['box']}") 