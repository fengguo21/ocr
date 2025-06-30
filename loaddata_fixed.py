from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import math
import json


class OCRDatasetFixed(Dataset):
    """修复版OCR数据集 - 正确实现CRAFT热力图生成"""
    
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
        if 'ocr' in item:
            for ocr_item in item['ocr']:
                if 'box' in ocr_item and 'word' in ocr_item:
                    box_coords = ocr_item['box']
                    word = ocr_item['word']
                    cls = ocr_item.get('cls', 0)
                    
                    # 将box坐标转换为四边形坐标
                    coords = self._parse_box_coords(box_coords)
                    annotations.append({
                        'coords': coords,
                        'text': word,
                        'cls': cls
                    })
        
        # 生成CRAFT训练用的热力图
        char_heatmap, link_heatmap = self._generate_heatmaps_fixed(image, annotations)
        
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
        if len(box_coords) == 4:
            x1, y1, x2, y2 = box_coords
            coords = [
                [x1, y1],  # 左上
                [x2, y1],  # 右上
                [x2, y2],  # 右下
                [x1, y2]   # 左下
            ]
        else:
            coords = []
            for i in range(0, len(box_coords), 2):
                if i + 1 < len(box_coords):
                    x = float(box_coords[i])
                    y = float(box_coords[i + 1])
                    coords.append([x, y])
        
        return np.array(coords)
    
    def _generate_heatmaps_fixed(self, image, annotations):
        """修复版：正确生成CRAFT热力图"""
        h, w = image.shape[:2]
        
        # 初始化热图
        char_heatmap = np.zeros((h, w), dtype=np.float32)
        link_heatmap = np.zeros((h, w), dtype=np.float32)
        
        # 按位置排序标注（从左到右，从上到下）
        sorted_annotations = self._sort_annotations_by_position(annotations)
        
        # 生成字符区域热图
        for ann in sorted_annotations:
            coords = ann['coords']
            self._generate_char_region_heatmap(char_heatmap, coords)
        
        # 生成相邻文本的链接热图
        self._generate_affinity_heatmap(link_heatmap, sorted_annotations, h, w)
        
        return char_heatmap, link_heatmap
    
    def _sort_annotations_by_position(self, annotations):
        """按位置排序标注（从左到右，从上到下）"""
        def get_center(coords):
            return (np.mean(coords[:, 0]), np.mean(coords[:, 1]))
        
        # 按y坐标分组（行）
        y_groups = {}
        for ann in annotations:
            center_x, center_y = get_center(ann['coords'])
            y_key = int(center_y // 50)  # 50像素为一行
            if y_key not in y_groups:
                y_groups[y_key] = []
            y_groups[y_key].append((ann, center_x, center_y))
        
        # 每行内按x坐标排序，行间按y坐标排序
        sorted_annotations = []
        for y_key in sorted(y_groups.keys()):
            row = sorted(y_groups[y_key], key=lambda x: x[1])  # 按x坐标排序
            sorted_annotations.extend([item[0] for item in row])
        
        return sorted_annotations
    
    def _generate_char_region_heatmap(self, heatmap, coords):
        """生成字符区域热图 - 覆盖整个文本框"""
        # 计算文本框的边界
        x_min, x_max = int(np.min(coords[:, 0])), int(np.max(coords[:, 0]))
        y_min, y_max = int(np.min(coords[:, 1])), int(np.max(coords[:, 1]))
        
        # 确保边界在图像范围内
        h, w = heatmap.shape
        x_min, x_max = max(0, x_min), min(w, x_max)
        y_min, y_max = max(0, y_min), min(h, y_max)
        
        if x_max <= x_min or y_max <= y_min:
            return
        
        # 计算文本框的中心和尺寸
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        
        # 生成覆盖整个文本框的高斯分布
        sigma_x = width / 4  # 水平方向的标准差
        sigma_y = height / 4  # 垂直方向的标准差
        
        # 为整个文本框区域生成高斯值
        for y in range(max(0, y_min - int(height/2)), 
                      min(h, y_max + int(height/2))):
            for x in range(max(0, x_min - int(width/2)), 
                          min(w, x_max + int(width/2))):
                
                # 计算距离中心的归一化距离
                dx = (x - center_x) / max(sigma_x, 1)
                dy = (y - center_y) / max(sigma_y, 1)
                
                # 生成椭圆高斯分布
                value = math.exp(-(dx**2 + dy**2) / 2)
                
                # 更新热图（取最大值）
                heatmap[y, x] = max(heatmap[y, x], value)
    
    def _generate_affinity_heatmap(self, heatmap, sorted_annotations, h, w):
        """生成相邻文本的链接热图"""
        for i in range(len(sorted_annotations) - 1):
            current = sorted_annotations[i]
            next_ann = sorted_annotations[i + 1]
            
            # 计算两个文本框的中心
            center1 = np.mean(current['coords'], axis=0)
            center2 = np.mean(next_ann['coords'], axis=0)
            
            # 检查是否为相邻文本（距离阈值）
            distance = np.linalg.norm(center2 - center1)
            if distance < 200:  # 调整此阈值
                self._draw_affinity_link(heatmap, center1, center2, h, w)
        
        # 同时处理同一行内的相邻文本
        self._generate_line_affinity(heatmap, sorted_annotations, h, w)
    
    def _draw_affinity_link(self, heatmap, point1, point2, h, w):
        """在两点间绘制链接热图"""
        x1, y1 = point1
        x2, y2 = point2
        
        # 计算连线上的点
        num_points = int(np.linalg.norm([x2-x1, y2-y1]) / 2)
        if num_points < 2:
            return
            
        for i in range(num_points):
            t = i / (num_points - 1)
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            
            if 0 <= x < w and 0 <= y < h:
                # 在连线周围生成高斯分布
                sigma = 5  # 链接的宽度
                for dy in range(-sigma*2, sigma*2+1):
                    for dx in range(-sigma*2, sigma*2+1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < w and 0 <= ny < h:
                            dist = math.sqrt(dx**2 + dy**2)
                            value = math.exp(-dist**2 / (2 * sigma**2))
                            heatmap[ny, nx] = max(heatmap[ny, nx], value)
    
    def _generate_line_affinity(self, heatmap, sorted_annotations, h, w):
        """生成同行文本的链接"""
        # 按行分组
        lines = self._group_by_lines(sorted_annotations)
        
        for line in lines:
            if len(line) > 1:
                for i in range(len(line) - 1):
                    center1 = np.mean(line[i]['coords'], axis=0)
                    center2 = np.mean(line[i+1]['coords'], axis=0)
                    
                    # 检查水平距离
                    if abs(center2[0] - center1[0]) < 150:  # 水平相邻
                        self._draw_affinity_link(heatmap, center1, center2, h, w)
    
    def _group_by_lines(self, annotations):
        """将标注按行分组"""
        lines = []
        used = set()
        
        for i, ann in enumerate(annotations):
            if i in used:
                continue
                
            center_y = np.mean(ann['coords'][:, 1])
            line = [ann]
            used.add(i)
            
            # 查找同一行的其他文本
            for j, other_ann in enumerate(annotations):
                if j in used:
                    continue
                    
                other_center_y = np.mean(other_ann['coords'][:, 1])
                if abs(other_center_y - center_y) < 30:  # 同一行的阈值
                    line.append(other_ann)
                    used.add(j)
            
            if line:
                # 按x坐标排序
                line.sort(key=lambda x: np.mean(x['coords'][:, 0]))
                lines.append(line)
        
        return lines
    
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


# 测试修复版本
if __name__ == "__main__":
    # 加载数据集
    ds = load_dataset("lansinuote/ocr_id_card")
    print("数据集键:", ds.keys())
    
    # 创建修复版数据集
    train_dataset = OCRDatasetFixed(ds, split='train')
    
    # 创建数据加载器
    data_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    
    # 测试数据加载
    print(f"修复版数据集大小: {len(train_dataset)}")
    
    for batch_idx, (images, char_heatmaps, link_heatmaps) in enumerate(data_loader):
        print(f"\n修复版批次 {batch_idx + 1}:")
        print(f"  图像形状: {images.shape}")
        print(f"  字符热图形状: {char_heatmaps.shape}")
        print(f"  链接热图形状: {link_heatmaps.shape}")
        print(f"  图像值范围: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  字符热图值范围: [{char_heatmaps.min():.3f}, {char_heatmaps.max():.3f}]")
        print(f"  链接热图值范围: [{link_heatmaps.min():.3f}, {link_heatmaps.max():.3f}]")
        
        if batch_idx >= 1:  # 只显示前2个批次
            break
    
    print("\n✅ 修复版数据加载器测试完成！") 