import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
from torchvision import models


class DoubleConv(nn.Module):
    """双重卷积块"""
    def __init__(self, in_ch, mid_ch, out_ch):
        super(DoubleConv, self).__init__()
        # 如果in_ch为0，表示不需要额外输入通道
        print(in_ch,mid_ch,out_ch,"in_ch,mid_ch,out_ch")
        if in_ch == 0:
            self.conv = nn.Sequential(
                nn.Conv2d(mid_ch, mid_ch, kernel_size=1),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class CRAFT(nn.Module):
    """CRAFT文本检测模型"""
    
    def __init__(self, pretrained=False, freeze=False):
        super(CRAFT, self).__init__()

        # 使用VGG16作为骨干网络
        vgg16 = models.vgg16(pretrained=pretrained)
        self.basenet = nn.ModuleList(list(vgg16.features.children()))

        # 上采样网络 - 修复通道数配置
        self.upconv1 = DoubleConv(0, 512, 256)    # conv5_3: 512 channels -> 256
        self.upconv2 = DoubleConv(512, 256, 128)  # upconv1: 256 + conv4_3: 512 = 768 -> 128
        self.upconv3 = DoubleConv(128, 256, 64)   # upconv2: 128 + conv3_3: 256 = 384 -> 64
        self.upconv4 = DoubleConv(0, 64, 32)      # upconv3: 64 -> 32

        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
            nn.Sigmoid()  # 🔧 添加Sigmoid激活函数，确保输出在[0,1]范围
        )

        self.init_weights()

        if freeze:
            for param in self.basenet.parameters():
                param.requires_grad = False

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # VGG16特征提取
        sources = []
        for k in range(len(self.basenet)):
            x = self.basenet[k](x)
            if k in [15, 22, 29]:  # conv3_3, conv4_3, conv5_3
                sources.append(x)
                # print(x.shape,"x")

        # 上采样和特征融合 - 修复通道数匹配
        # conv5_3: 512 channels，不需要拼接
        y = self.upconv1(sources[2])  # 直接使用conv5_3  sources[2] 512 -> 256

        # 上采样到conv4_3的尺寸并拼接
        y = F.interpolate(y, size=sources[1].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[1]], dim=1)  # 256 + 512 = 768 -> 使用DoubleConv调整
        y = self.upconv2(y)

        # 上采样到conv3_3的尺寸并拼接
        y = F.interpolate(y, size=sources[0].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[0]], dim=1)  # 128 + 256 = 384 -> 使用DoubleConv调整
        y = self.upconv3(y)

        # 最终上采样
        y = F.interpolate(y, size=(sources[0].size(2)*2, sources[0].size(3)*2), mode='bilinear', align_corners=False)

        feature = self.upconv4(y)

        y = self.conv_cls(feature)

        return y.permute(0, 2, 3, 1), feature


class TextDetector:
    """文本检测器封装类"""
    
    def __init__(self, model_path=None, device='cpu', use_pretrained=True):
        self.device = device
        self.net = CRAFT(pretrained=use_pretrained, freeze=False).to(device)
        
        if model_path and os.path.exists(model_path):
            print(f"正在加载自定义模型权重: {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            
            # 处理不同的保存格式
            if 'model_state_dict' in checkpoint:
                # 完整的checkpoint格式
                self.net.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ 加载训练好的模型 - Epoch: {checkpoint.get('epoch', 'unknown')}")
                print(f"训练损失: {checkpoint.get('train_loss', 'unknown'):.4f}" if checkpoint.get('train_loss') else "")
            else:
                # 只有模型权重的格式
                self.net.load_state_dict(checkpoint)
                print("✅ 模型权重加载成功")
        elif model_path:
            print(f"⚠️ 模型文件不存在: {model_path}")
            if use_pretrained:
                print("使用ImageNet预训练权重")
            else:
                print("使用随机初始化权重")
        
        self.net.eval()
        
    def detect_text(self, image, text_threshold=0.7, link_threshold=0.4, 
                   low_text=0.4, poly=False):
        """
        检测图像中的文本区域
        """
        # 图像预处理
        img_resized, target_ratio, size_heatmap = self.resize_aspect_ratio(
            image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
        ratio_h = ratio_w = 1 / target_ratio

        # 转换为tensor
        x = self.normalize_mean_variance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        # 推理
        with torch.no_grad():
            y, _ = self.net(x)

        # 后处理
        score_text = y[0, :, :, 0].cpu().numpy()
        score_link = y[0, :, :, 1].cpu().numpy()

        # 获取文本框
        boxes, polys = self.get_det_boxes(
            score_text, score_link, text_threshold, link_threshold, low_text, poly)

        # 调整坐标
        boxes = self.adjust_result_coordinates(boxes, ratio_w, ratio_h)
        polys = self.adjust_result_coordinates(polys, ratio_w, ratio_h)
        print(boxes,"boxes")
        print(polys,"polys")
        

        return boxes, polys

    def resize_aspect_ratio(self, img, square_size, interpolation, mag_ratio=1):
        height, width, channel = img.shape

        # 计算目标尺寸
        target_size = mag_ratio * max(height, width)

        if target_size > square_size:
            target_size = square_size
        
        ratio = target_size / max(height, width)    

        target_h, target_w = int(height * ratio), int(width * ratio)
        proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

        # 填充到正方形
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

    def normalize_mean_variance(self, in_img, mean=(0.485, 0.456, 0.406), 
                              variance=(0.229, 0.224, 0.225)):
        # 应该在0~1范围内
        img = in_img.copy().astype(np.float32)

        img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
        img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
        return img

    def get_det_boxes(self, textmap, linkmap, text_threshold, link_threshold, low_text, poly=False):
        # 实现文本框检测算法
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

            # 获取边界框
            if np.max(textmap[labels == k]) < text_threshold:
                continue

            # 获取边界框坐标
            segmap = np.zeros(textmap.shape, dtype=np.uint8)
            segmap[labels == k] = 255
            segmap[np.logical_and(link_score == 1, text_score == 0)] = 0

            x, y, w, h = cv2.boundingRect(segmap)
            niter = int(min(w, h) * 0.03)
            sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
            
            if sx < 0:
                sx = 0
            if sy < 0:
                sy = 0
            if ex >= img_w:
                ex = img_w
            if ey >= img_h:
                ey = img_h
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
            segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

            # 获取轮廓 - 兼容不同OpenCV版本
            contour_info = cv2.findContours(segmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contour_info) == 3:
                # OpenCV 3.x 返回 (image, contours, hierarchy)
                contours = contour_info[1]
            else:
                # OpenCV 4.x 返回 (contours, hierarchy)
                contours = contour_info[0]
                
            if len(contours) == 0:
                continue
            
            contour = contours[0]
            
            # 生成边界框
            if poly:
                epsilon = 0.002 * cv2.arcLength(contour, True)
                points = cv2.approxPolyDP(contour, epsilon, True)
                points = np.array(points).reshape((-1, 2))
            else:
                rect = cv2.minAreaRect(contour)
                points = cv2.boxPoints(rect)

            det.append(points)

        return det, det

    def adjust_result_coordinates(self, polys, ratio_w, ratio_h):
        if len(polys) > 0:
            polys = np.array(polys)
            for k in range(len(polys)):
                if polys[k] is not None:
                    polys[k] *= (ratio_w * 2, ratio_h * 2)
        return polys 