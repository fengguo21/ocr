import cv2
import numpy as np
from PIL import Image, ImageEnhance
import albumentations as A
from typing import Tuple, List, Optional


class IDCardPreprocessor:
    """身份证图像预处理器"""
    
    def __init__(self):
        self.target_size = (640, 480)  # 身份证标准尺寸比例
        
        # 定义数据增强管道
        self.train_transform = A.Compose([
            A.OneOf([
                A.GaussNoise(noise_scale_factor=0.5, p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.5),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            ], p=0.4),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.5),
                A.ToGray(p=0.2),
            ], p=0.3),
            A.Affine(rotate=(-5, 5), scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), p=0.5),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
        ])
        
        self.test_transform = A.Compose([
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        ])

    def detect_id_card(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        检测并提取身份证区域
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 边缘检测
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # 查找轮廓 - 兼容不同OpenCV版本
        contour_info = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contour_info) == 3:
            # OpenCV 3.x 返回 (image, contours, hierarchy)
            contours = contour_info[1]
        else:
            # OpenCV 4.x 返回 (contours, hierarchy)
            contours = contour_info[0]
        
        # 过滤轮廓，寻找矩形
        card_contour = None
        max_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:  # 过滤太小的区域
                continue
                
            # 近似轮廓
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 检查是否为四边形
            if len(approx) == 4 and area > max_area:
                max_area = area
                card_contour = approx
        
        if card_contour is not None:
            # 透视变换提取身份证
            return self.perspective_transform(image, card_contour)
        
        return None

    def perspective_transform(self, image: np.ndarray, contour: np.ndarray) -> np.ndarray:
        """
        透视变换提取身份证区域
        """
        # 重新排序点的顺序
        pts = contour.reshape(4, 2).astype(np.float32)
        
        # 计算每个点到原点的距离
        rect = np.zeros((4, 2), dtype=np.float32)
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # 左上角
        rect[2] = pts[np.argmax(s)]  # 右下角
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # 右上角
        rect[3] = pts[np.argmax(diff)]  # 左下角
        
        # 计算新图像的宽高
        width_a = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
        width_b = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
        max_width = max(int(width_a), int(width_b))
        
        height_a = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
        height_b = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
        max_height = max(int(height_a), int(height_b))
        
        # 目标点
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)
        
        # 透视变换
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (max_width, max_height))
        
        return warped

    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        增强图像质量
        """
        # 转换为PIL图像进行增强
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 亮度增强
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(1.2)
        
        # 对比度增强
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.3)
        
        # 锐度增强
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(1.2)
        
        # 转换回OpenCV格式
        enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return enhanced

    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像去噪
        """
        # 双边滤波去噪
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # 非局部均值去噪
        if len(image.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(denoised, None, 10, 10, 7, 21)
        else:
            denoised = cv2.fastNlMeansDenoising(denoised, None, 10, 7, 21)
        
        return denoised

    def correct_skew(self, image: np.ndarray) -> np.ndarray:
        """
        纠正图像倾斜
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # 霍夫变换检测直线
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            angles = []
            # 修复霍夫变换结果的解包问题
            for line in lines[:10]:  # 只考虑前10条线
                if len(line) >= 2:
                    rho, theta = line[0], line[1]
                elif len(line[0]) >= 2:
                    rho, theta = line[0][0], line[0][1]
                else:
                    continue
                    
                angle = theta * 180 / np.pi
                if angle < 45:
                    angles.append(angle)
                elif angle > 135:
                    angles.append(angle - 180)
            
            if angles:
                # 计算平均角度
                median_angle = np.median(angles)
                
                # 旋转图像
                if abs(median_angle) > 0.5:  # 只有角度大于0.5度才旋转
                    (h, w) = image.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, 
                                         borderMode=cv2.BORDER_REPLICATE)
        
        return image

    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
        """
        调整图像尺寸，保持纵横比
        """
        if target_size is None:
            target_size = self.target_size
            
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # 计算缩放比例
        scale = min(target_w / w, target_h / h)
        
        # 计算新尺寸
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 调整尺寸
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 填充到目标尺寸
        top = (target_h - new_h) // 2
        bottom = target_h - new_h - top
        left = (target_w - new_w) // 2
        right = target_w - new_w - left
        
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                  cv2.BORDER_CONSTANT, value=[255, 255, 255])
        
        return padded

    def preprocess_for_ocr(self, image: np.ndarray, training: bool = False) -> np.ndarray:
        """
        OCR预处理管道
        """
        # 1. 身份证区域检测和提取
        card_region = self.detect_id_card(image)
        if card_region is not None:
            image = card_region
        
        # 2. 图像增强
        image = self.enhance_image(image)
        
        # 3. 去噪
        image = self.denoise_image(image)
        
        # 4. 倾斜纠正
        image = self.correct_skew(image)
        
        # 5. 调整尺寸
        image = self.resize_image(image)
        
        # 6. 数据增强（仅训练时）
        if training:
            image = self.train_transform(image=image)['image']
        else:
            image = self.test_transform(image=image)['image']
        
        return image

    def extract_text_regions(self, image: np.ndarray) -> List[np.ndarray]:
        """
        提取文本区域
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 形态学操作提取文本区域
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓 - 兼容不同OpenCV版本
        contour_info = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contour_info) == 3:
            # OpenCV 3.x 返回 (image, contours, hierarchy)
            contours = contour_info[1]
        else:
            # OpenCV 4.x 返回 (contours, hierarchy)
            contours = contour_info[0]
        
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 50 and h > 15:  # 过滤太小的区域
                text_region = image[y:y+h, x:x+w]
                text_regions.append(text_region)
        
        return text_regions

    def preprocess_text_region(self, region: np.ndarray) -> np.ndarray:
        """
        预处理单个文本区域
        """
        # 转换为灰度图
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region
        
        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 形态学操作去除噪点
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned 