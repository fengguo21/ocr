from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import math
import json


class OCRDataset(Dataset):
    """适配带有box、cls、word信息的OCR数据集"""
    
    def __init__(self, hf_dataset, split='train', input_size=512, target_size=256):
        self.dataset = hf_dataset[split]
        self.input_size = input_size
        self.target_size = target_size
        
        # 图像变换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # 获取图像
        image = item['image']
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # 获取标注信息
        annotations = []
        if 'box' in item and 'word' in item:
            box_coords = item['box']
            word = item['word']
            cls = item.get('cls', 0)
            
            # 将box坐标转换为四边形坐标
            coords = self._parse_box_coords(box_coords)
            annotations.append({
                'coords': coords,
                'text': word,
                'cls': cls
            })
        
        # 生成CRAFT训练用的热力图
        char_heatmap, link_heatmap = self._generate_heatmaps(image, annotations)
        
        # 调整尺寸
        image, char_heatmap, link_heatmap = self._resize_data(
            image, char_heatmap, link_heatmap
        )
        
        # 转换为tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        char_heatmap = torch.from_numpy(char_heatmap).float()
        link_heatmap = torch.from_numpy(link_heatmap).float()
        
        return image, char_heatmap, link_heatmap
    
    def _parse_box_coords(self, box_coords):
        """解析边界框坐标"""
        # 假设box_coords是[x1, y1, x2, y2, x3, y3, x4, y4]格式
        coords = []
        for i in range(0, len(box_coords), 2):
            if i + 1 < len(box_coords):
                x = float(box_coords[i])
                y = float(box_coords[i + 1])
                coords.append([x, y])
        
        # 如果只有两个点（矩形），扩展为四个点
        if len(coords) == 2:
            x1, y1 = coords[0]
            x2, y2 = coords[1]
            coords = [
                [x1, y1],  # 左上
                [x2, y1],  # 右上
                [x2, y2],  # 右下
                [x1, y2]   # 左下
            ]
        
        return np.array(coords)
    
    def _generate_heatmaps(self, image, annotations):
        """生成字符级和链接级热图"""
        h, w = image.shape[:2]
        
        # 初始化热图
        char_heatmap = np.zeros((h, w), dtype=np.float32)
        link_heatmap = np.zeros((h, w), dtype=np.float32)
        
        for ann in annotations:
            coords = ann['coords']
            
            # 生成字符级热图
            self._generate_char_heatmap(char_heatmap, coords)
            
            # 生成链接级热图
            self._generate_link_heatmap(link_heatmap, coords)
        
        return char_heatmap, link_heatmap
    
    def _generate_char_heatmap(self, heatmap, coords):
        """生成字符级热图"""
        # 计算文本框的中心和尺寸
        center_x = np.mean(coords[:, 0])
        center_y = np.mean(coords[:, 1])
        
        # 计算文本框的大小
        width = np.linalg.norm(coords[1] - coords[0])
        height = np.linalg.norm(coords[2] - coords[1]) if len(coords) > 2 else width
        
        # 生成高斯热图
        sigma = min(width, height) / 6
        if sigma > 0:
            self._add_gaussian_heatmap(heatmap, center_x, center_y, sigma)
    
    def _generate_link_heatmap(self, heatmap, coords):
        """生成链接级热图"""
        # 在文本框边缘生成连接热图
        for i in range(len(coords)):
            j = (i + 1) % len(coords)
            
            start_x, start_y = coords[i]
            end_x, end_y = coords[j]
            
            # 在边的中点生成连接
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2
            
            # 计算边的长度作为sigma
            edge_length = np.linalg.norm([end_x - start_x, end_y - start_y])
            sigma = edge_length / 8
            
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
    
    def _resize_data(self, image, char_heatmap, link_heatmap):
        """调整数据尺寸"""
        h, w = image.shape[:2]
        
        # 计算缩放比例
        scale = self.input_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # 调整图像尺寸
        image = cv2.resize(image, (new_w, new_h))
        
        # 填充图像到input_size
        pad_h = self.input_size - new_h
        pad_w = self.input_size - new_w
        
        image = cv2.copyMakeBorder(
            image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
        )
        
        # 调整热图尺寸到target_size
        target_scale = self.target_size / max(h, w)
        target_new_h, target_new_w = int(h * target_scale), int(w * target_scale)
        
        char_heatmap = cv2.resize(char_heatmap, (target_new_w, target_new_h))
        link_heatmap = cv2.resize(link_heatmap, (target_new_w, target_new_h))
        
        # 填充热图到target_size
        target_pad_h = self.target_size - target_new_h
        target_pad_w = self.target_size - target_new_w
        
        char_heatmap = cv2.copyMakeBorder(
            char_heatmap, 0, target_pad_h, 0, target_pad_w, cv2.BORDER_CONSTANT, value=0
        )
        link_heatmap = cv2.copyMakeBorder(
            link_heatmap, 0, target_pad_h, 0, target_pad_w, cv2.BORDER_CONSTANT, value=0
        )
        
        return image, char_heatmap, link_heatmap


# 加载数据集
ds = load_dataset("lansinuote/ocr_id_card")
print("数据集键:", ds.keys())

# 创建自定义数据集
train_dataset = OCRDataset(ds, split='train')

# 创建数据加载器
data_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)

# 测试数据加载
print(f"数据集大小: {len(train_dataset)}")

for batch_idx, (images, char_heatmaps, link_heatmaps) in enumerate(data_loader):
    print(f"\n批次 {batch_idx + 1}:")
    print(f"  图像形状: {images.shape}")
    print(f"  字符热图形状: {char_heatmaps.shape}")
    print(f"  链接热图形状: {link_heatmaps.shape}")
    print(f"  图像值范围: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  字符热图值范围: [{char_heatmaps.min():.3f}, {char_heatmaps.max():.3f}]")
    print(f"  链接热图值范围: [{link_heatmaps.min():.3f}, {link_heatmaps.max():.3f}]")
    
    if batch_idx >= 2:  # 只显示前3个批次
        break

print("\n✅ 数据加载器设置完成！可以用于CRAFT模型训练。") 