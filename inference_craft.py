"""
CRAFT文本检测推理脚本
使用训练好的模型权重进行文本检测
"""

import torch
import cv2
import numpy as np
from PIL import Image
import argparse
import os
import matplotlib.pyplot as plt

from models.text_detection import CRAFT, TextDetector


class TrainedCRAFTDetector:
    """使用训练好的CRAFT模型进行文本检测"""
    
    def __init__(self, model_path, device='cpu'):
        self.device = device
        
        # 创建模型
        self.model = CRAFT(pretrained=False, freeze=False).to(device)
        
        # 加载训练好的权重
        if model_path and os.path.exists(model_path):
            print(f"正在加载模型权重: {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            
            # 处理不同的保存格式
            if 'model_state_dict' in checkpoint:
                # 完整的checkpoint格式
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"模型训练epoch: {checkpoint.get('epoch', 'unknown')}")
                print(f"训练损失: {checkpoint.get('train_loss', 'unknown')}")
            else:
                # 只有模型权重的格式
                self.model.load_state_dict(checkpoint)
            
            print("✅ 模型权重加载成功")
        else:
            print(f"⚠️ 模型文件不存在: {model_path}")
            print("使用预训练权重或随机初始化的权重")
        
        self.model.eval()
    
    def detect_text(self, image_path, text_threshold=0.7, link_threshold=0.4, 
                   low_text=0.4, output_dir='results'):
        """
        检测图像中的文本
        
        Args:
            image_path: 输入图像路径
            text_threshold: 文本检测阈值
            link_threshold: 链接检测阈值
            low_text: 低文本阈值
            output_dir: 输出目录
        """
        # 读取图像
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path
        
        # 图像预处理
        img_resized, target_ratio, size_heatmap = self._resize_aspect_ratio(
            image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
        ratio_h = ratio_w = 1 / target_ratio

        # 转换为tensor
        x = self._normalize_mean_variance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        # 推理
        with torch.no_grad():
            y, _ = self.model(x)

        # 后处理
        score_text = y[0, :, :, 0].cpu().numpy()
        score_link = y[0, :, :, 1].cpu().numpy()

        # 获取文本框
        boxes, polys = self._get_det_boxes(
            score_text, score_link, text_threshold, link_threshold, low_text)

        # 调整坐标
        boxes = self._adjust_result_coordinates(boxes, ratio_w, ratio_h)
        polys = self._adjust_result_coordinates(polys, ratio_w, ratio_h)

        # 保存结果
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self._save_results(image, boxes, score_text, score_link, 
                             output_dir, os.path.basename(image_path) if isinstance(image_path, str) else 'result')

        return boxes, polys, score_text, score_link
    
    def _resize_aspect_ratio(self, img, square_size, interpolation, mag_ratio=1):
        """调整图像尺寸保持宽高比"""
        height, width, channel = img.shape

        target_size = mag_ratio * max(height, width)
        if target_size > square_size:
            target_size = square_size
        
        ratio = target_size / max(height, width)    
        target_h, target_w = int(height * ratio), int(width * ratio)
        proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

        # 填充到32的倍数
        target_h32, target_w32 = target_h, target_w
        if target_h % 32 != 0:
            target_h32 = target_h + (32 - target_h % 32)
        if target_w % 32 != 0:
            target_w32 = target_w + (32 - target_w % 32)
        resized = np.zeros((target_h32, target_w32, channel), dtype=np.uint8)
        resized[0:target_h, 0:target_w, :] = proc
        target_h, target_w = target_h32, target_w32

        size_heatmap = (int(target_w/2), int(target_h/2))
        return resized, ratio, size_heatmap

    def _normalize_mean_variance(self, in_img, mean=(0.485, 0.456, 0.406), 
                              variance=(0.229, 0.224, 0.225)):
        """图像归一化"""
        img = in_img.copy().astype(np.float32)
        img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
        img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
        return img

    def _get_det_boxes(self, textmap, linkmap, text_threshold, link_threshold, low_text):
        """从热力图提取文本框"""
        img_h, img_w = textmap.shape

        ret, text_score = cv2.threshold(textmap, low_text, 1, cv2.THRESH_BINARY)
        ret, link_score = cv2.threshold(linkmap, link_threshold, 1, cv2.THRESH_BINARY)

        text_score_comb = np.clip(text_score + link_score, 0, 1)
        nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            text_score_comb.astype(np.uint8), connectivity=4)

        det = []
        for k in range(1, nLabels):
            # 过滤小区域
            size = stats[k, cv2.CC_STAT_AREA]
            if size < 10:
                continue

            # 检查文本得分
            if np.max(textmap[labels == k]) < text_threshold:
                continue

            # 获取边界框
            segmap = np.zeros(textmap.shape, dtype=np.uint8)
            segmap[labels == k] = 255
            segmap[np.logical_and(link_score == 1, text_score == 0)] = 0

            x, y, w, h = cv2.boundingRect(segmap)
            niter = int(min(w, h) * 0.03)
            sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
            
            # 边界检查
            sx, sy = max(0, sx), max(0, sy)
            ex, ey = min(img_w, ex), min(img_h, ey)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
            segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

            # 获取轮廓
            contour_info = cv2.findContours(segmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = contour_info[1] if len(contour_info) == 3 else contour_info[0]
                
            if len(contours) == 0:
                continue
            
            contour = contours[0]
            rect = cv2.minAreaRect(contour)
            points = cv2.boxPoints(rect)
            det.append(points)

        return det, det

    def _adjust_result_coordinates(self, polys, ratio_w, ratio_h):
        """调整坐标到原图尺寸"""
        if len(polys) > 0:
            polys = np.array(polys)
            for k in range(len(polys)):
                if polys[k] is not None:
                    polys[k] *= (ratio_w * 2, ratio_h * 2)
        return polys

    def _save_results(self, image, boxes, score_text, score_link, output_dir, filename):
        """保存检测结果"""
        # 保存带框的图像
        result_img = image.copy()
        for box in boxes:
            if box is not None:
                box = box.astype(np.int32)
                cv2.polylines(result_img, [box], True, (0, 255, 0), 2)
        
        # 创建可视化
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 原图
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('原图')
        axes[0, 0].axis('off')
        
        # 检测结果
        axes[0, 1].imshow(result_img)
        axes[0, 1].set_title(f'检测结果 ({len(boxes)} 个文本区域)')
        axes[0, 1].axis('off')
        
        # 字符热图
        axes[1, 0].imshow(score_text, cmap='hot')
        axes[1, 0].set_title('字符热图')
        axes[1, 0].axis('off')
        
        # 链接热图
        axes[1, 1].imshow(score_link, cmap='hot')
        axes[1, 1].set_title('链接热图')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'detection_{filename}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 检测结果已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='CRAFT文本检测推理')
    parser.add_argument('--model_path', type=str, help='训练好的模型权重路径')
    parser.add_argument('--image_path', type=str, required=True, help='输入图像路径')
    parser.add_argument('--output_dir', type=str, default='detection_results', help='输出目录')
    parser.add_argument('--text_threshold', type=float, default=0.7, help='文本检测阈值')
    parser.add_argument('--link_threshold', type=float, default=0.4, help='链接检测阈值')
    parser.add_argument('--device', type=str, default='cpu', help='设备 (cpu/cuda)')
    
    args = parser.parse_args()
    
    # 检查设备
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA不可用，使用CPU")
        device = 'cpu'
    
    print(f"使用设备: {device}")
    
    # 创建检测器
    detector = TrainedCRAFTDetector(args.model_path, device)
    
    # 进行检测
    print(f"正在检测图像: {args.image_path}")
    boxes, polys, score_text, score_link = detector.detect_text(
        args.image_path,
        text_threshold=args.text_threshold,
        link_threshold=args.link_threshold,
        output_dir=args.output_dir
    )
    
    print(f"🎯 检测完成！找到 {len(boxes)} 个文本区域")


if __name__ == "__main__":
    main() 