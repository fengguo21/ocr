#!/usr/bin/env python3
"""
训练数据准备脚本
用于生成CRNN和CRAFT模型的训练数据
"""

import os
import cv2
import numpy as np
import json
from typing import List, Tuple, Dict
import argparse
from pathlib import Path


class TrainingDataPreparer:
    """训练数据准备器"""
    
    def __init__(self):
        self.crnn_height = 32  # CRNN模型固定高度
        
    def prepare_crnn_data(self, input_dir: str, output_dir: str):
        """
        准备CRNN训练数据
        
        Args:
            input_dir: 包含身份证图像的目录
            output_dir: 输出训练数据的目录
        """
        print("🔧 准备CRNN训练数据...")
        
        # 创建输出目录
        images_dir = Path(output_dir) / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        labels_file = Path(output_dir) / "labels.txt"
        
        # 身份证常见文本样本
        sample_texts = [
            "任亚坤", "张三", "李四", "王五", "赵六", "钱七", "孙八", "周九", "吴十",
            "男", "女",
            "汉族", "满族", "蒙古族", "回族", "藏族", "维吾尔族", "苗族", "彝族",
            "1988年12月4日", "1990年1月1日", "1985年3月15日", "1992年8月20日",
            "河南省商丘市睢阳区", "北京市朝阳区", "上海市浦东新区", "广东省深圳市",
            "411403198812047555", "110101199001010000", "320101198501010000",
            "商丘市公安局睢阳分局", "北京市公安局朝阳分局", "上海市公安局浦东分局",
            "2016.01.28-2036.01.28", "2010.01.01-2030.01.01", "2015.06.10-2035.06.10"
        ]
        
        with open(labels_file, 'w', encoding='utf-8') as f:
            for idx, text in enumerate(sample_texts):
                # 生成文本图像
                image = self._generate_text_image(text)
                if image is not None:
                    image_name = f"{idx:05d}.jpg"
                    image_path = images_dir / image_name
                    cv2.imwrite(str(image_path), image)
                    f.write(f"{image_name} {text}\n")
                    print(f"  生成: {image_name} -> {text}")
        
        print(f"✅ CRNN训练数据已保存到: {output_dir}")
        print(f"   - 图像数量: {len(sample_texts)}")
        print(f"   - 标签文件: {labels_file}")
    
    def _generate_text_image(self, text: str) -> np.ndarray:
        """
        生成文本图像
        
        Args:
            text: 要生成的文本
            
        Returns:
            生成的图像数组
        """
        # 字体配置
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # 获取文本尺寸
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # 计算图像尺寸
        padding = 10
        img_width = text_width + 2 * padding
        img_height = self.crnn_height
        
        # 创建白色背景图像
        image = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
        
        # 计算文本位置（居中）
        x = (img_width - text_width) // 2
        y = (img_height + text_height) // 2
        
        # 绘制文本
        cv2.putText(image, text, (x, y), font, font_scale, (0, 0, 0), thickness)
        
        return image
    
    def prepare_craft_data(self, input_dir: str, output_dir: str):
        """
        准备CRAFT训练数据
        
        Args:
            input_dir: 包含身份证图像的目录
            output_dir: 输出训练数据的目录
        """
        print("🔧 准备CRAFT训练数据...")
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 处理输入图像
        input_path = Path(input_dir)
        image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.jpeg")) + list(input_path.glob("*.png"))
        
        for idx, image_file in enumerate(image_files):
            print(f"  处理图像: {image_file.name}")
            
            # 读取图像
            image = cv2.imread(str(image_file))
            if image is None:
                continue
            
            # 复制图像到输出目录
            output_image_name = f"img_{idx:03d}.jpg"
            output_image_path = output_path / output_image_name
            cv2.imwrite(str(output_image_path), image)
            
            # 生成对应的标注文件
            gt_file_name = f"gt_img_{idx:03d}.txt"
            gt_file_path = output_path / gt_file_name
            
            # 模拟标注数据（实际应用中需要人工标注）
            sample_annotations = self._generate_sample_annotations(image.shape)
            
            with open(gt_file_path, 'w', encoding='utf-8') as f:
                for ann in sample_annotations:
                    f.write(f"{ann}\n")
            
            print(f"    生成标注: {gt_file_name}")
        
        print(f"✅ CRAFT训练数据已保存到: {output_dir}")
        print(f"   - 图像数量: {len(image_files)}")
    
    def _generate_sample_annotations(self, image_shape: Tuple[int, int, int]) -> List[str]:
        """
        生成示例标注数据（实际应用中需要人工标注）
        
        Args:
            image_shape: 图像尺寸
            
        Returns:
            标注数据列表
        """
        h, w, _ = image_shape
        
        # 模拟身份证上的文本区域
        annotations = [
            f"100,50,200,50,200,80,100,80,姓名",
            f"100,90,150,90,150,120,100,120,性别", 
            f"100,130,180,130,180,160,100,160,民族",
            f"100,170,250,170,250,200,100,200,出生日期",
            f"100,210,400,210,400,280,100,280,住址",
            f"100,290,350,290,350,320,100,320,身份证号码"
        ]
        
        # 调整坐标到图像范围内
        adjusted_annotations = []
        for ann in annotations:
            parts = ann.split(',')
            coords = [int(x) for x in parts[:8]]
            text = parts[8]
            
            # 确保坐标在图像范围内
            for i in range(0, 8, 2):
                coords[i] = min(coords[i], w - 1)  # x坐标
            for i in range(1, 8, 2):
                coords[i] = min(coords[i], h - 1)  # y坐标
            
            adjusted_ann = ','.join([str(c) for c in coords] + [text])
            adjusted_annotations.append(adjusted_ann)
        
        return adjusted_annotations
    
    def create_dataset_info(self, output_dir: str):
        """
        创建数据集信息文件
        
        Args:
            output_dir: 输出目录
        """
        dataset_info = {
            "name": "身份证OCR训练数据集",
            "description": "用于训练身份证识别模型的数据集",
            "version": "1.0",
            "models": {
                "CRNN": {
                    "task": "文本识别",
                    "input": "文本区域图像",
                    "output": "识别的文字",
                    "image_height": 32,
                    "data_format": "images/ + labels.txt"
                },
                "CRAFT": {
                    "task": "文本检测", 
                    "input": "完整场景图像",
                    "output": "文本框坐标",
                    "data_format": "images/ + gt_*.txt"
                }
            },
            "charset": "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ中华人民共和国身份证姓名性别民族出生年月日住址公民号码签发机关有效期限长至汉族男女省市县区街道路号楼室派出所公安局厅",
            "fields": [
                "姓名", "性别", "民族", "出生", "住址", "公民身份号码", "签发机关", "有效期限"
            ]
        }
        
        info_file = Path(output_dir) / "dataset_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
        print(f"📋 数据集信息已保存到: {info_file}")


def main():
    parser = argparse.ArgumentParser(description='训练数据准备工具')
    parser.add_argument('--input', type=str, help='输入图像目录')
    parser.add_argument('--output', type=str, default='training_data', help='输出目录')
    parser.add_argument('--model', type=str, choices=['crnn', 'craft', 'both'], 
                       default='both', help='准备哪个模型的数据')
    
    args = parser.parse_args()
    
    preparer = TrainingDataPreparer()
    
    if args.model in ['crnn', 'both']:
        # 准备CRNN数据
        crnn_output = Path(args.output) / "crnn_data"
        preparer.prepare_crnn_data(args.input or ".", str(crnn_output))
    
    if args.model in ['craft', 'both']:
        # 准备CRAFT数据
        craft_output = Path(args.output) / "craft_data"
        preparer.prepare_craft_data(args.input or ".", str(craft_output))
    
    # 创建数据集信息
    preparer.create_dataset_info(args.output)
    
    print("\n🎉 训练数据准备完成！")
    print(f"📁 输出目录: {args.output}")
    print("\n📖 使用说明:")
    print("1. CRNN训练数据位于: crnn_data/")
    print("   - images/: 文本图像文件")
    print("   - labels.txt: 对应的文本标签")
    print("2. CRAFT训练数据位于: craft_data/")
    print("   - img_*.jpg: 场景图像")
    print("   - gt_*.txt: 对应的标注文件")
    print("3. 数据集信息: dataset_info.json")


if __name__ == "__main__":
    main() 