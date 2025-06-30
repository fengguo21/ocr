"""
身份证识别系统演示脚本
展示不同OCR方法的使用和效果对比
"""

import os
import sys
import cv2
import numpy as np
import json
import time
from typing import Dict, List
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

from main import IDCardRecognizer


class OCRDemo:
    """OCR演示类"""
    
    def __init__(self):
        self.methods = ['paddleocr', 'easyocr', 'hybrid']
        self.recognizers = {}
        
        # 初始化各种识别器
        for method in self.methods:
            try:
                config = {'ocr_method': method}
                self.recognizers[method] = IDCardRecognizer(config)
                print(f"✓ {method} 初始化成功")
            except Exception as e:
                print(f"✗ {method} 初始化失败: {str(e)}")
    
    def compare_methods(self, image_path: str):
        """比较不同OCR方法的效果"""
        print(f"\n🔍 比较不同OCR方法处理: {image_path}")
        print("=" * 60)
        
        results = {}
        times = {}
        
        for method, recognizer in self.recognizers.items():
            print(f"\n正在使用 {method} 识别...")
            
            start_time = time.time()
            try:
                result = recognizer.recognize_image(image_path)
                end_time = time.time()
                
                results[method] = result
                times[method] = end_time - start_time
                
                print(f"✓ {method} 完成，耗时: {times[method]:.2f}秒")
                
            except Exception as e:
                print(f"✗ {method} 失败: {str(e)}")
                results[method] = None
                times[method] = 0
        
        # 显示结果对比
        self._display_comparison(results, times)
        
        return results, times
    
    def _display_comparison(self, results: Dict, times: Dict):
        """显示结果对比"""
        print("\n📊 识别结果对比:")
        print("=" * 80)
        
        # 字段名称
        fields = ['姓名', '性别', '民族', '出生', '住址', '公民身份号码', '签发机关', '有效期限']
        
        # 表头
        print(f"{'字段':<12}", end='')
        for method in self.methods:
            if method in results and results[method]:
                print(f"{method:<20}", end='')
        print()
        
        print("-" * 80)
        
        # 各字段结果
        for field in fields:
            print(f"{field:<12}", end='')
            for method in self.methods:
                if method in results and results[method]:
                    value = results[method].get(field, '')
                    # 限制显示长度
                    display_value = value[:18] + '...' if len(value) > 18 else value
                    print(f"{display_value:<20}", end='')
            print()
        
        # 处理时间对比
        print("\n⏱️ 处理时间对比:")
        print("-" * 40)
        for method, time_cost in times.items():
            if time_cost > 0:
                print(f"{method:<15}: {time_cost:.2f}秒")
    
    def accuracy_test(self, test_data: List[Dict]):
        """准确率测试"""
        print("\n🎯 准确率测试")
        print("=" * 60)
        
        accuracy_stats = {}
        
        for method in self.methods:
            if method not in self.recognizers:
                continue
                
            print(f"\n测试 {method}:")
            recognizer = self.recognizers[method]
            
            correct_fields = {'姓名': 0, '性别': 0, '公民身份号码': 0}
            total_samples = len(test_data)
            
            for i, sample in enumerate(test_data):
                image_path = sample['image']
                ground_truth = sample['labels']
                
                try:
                    result = recognizer.recognize_image(image_path)
                    
                    # 检查准确率
                    for field in correct_fields:
                        if field in ground_truth and field in result:
                            if result[field] == ground_truth[field]:
                                correct_fields[field] += 1
                    
                    print(f"  样本 {i+1}/{total_samples} 完成")
                    
                except Exception as e:
                    print(f"  样本 {i+1} 失败: {str(e)}")
            
            # 计算准确率
            accuracy = {}
            for field, correct_count in correct_fields.items():
                accuracy[field] = (correct_count / total_samples) * 100
            
            accuracy_stats[method] = accuracy
            
            print(f"  {method} 结果:")
            for field, acc in accuracy.items():
                print(f"    {field}: {acc:.1f}%")
        
        return accuracy_stats
    
    def visualize_results(self, image_path: str, results: Dict):
        """可视化识别结果"""
        # 读取原图
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('身份证识别结果对比', fontsize=16, fontweight='bold')
        
        # 显示原图
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('原始图像')
        axes[0, 0].axis('off')
        
        # 显示各方法结果
        method_positions = [(0, 1), (1, 0), (1, 1)]
        
        for i, method in enumerate(self.methods[:3]):
            if method in results and results[method]:
                row, col = method_positions[i]
                
                # 创建结果文本
                result_text = f"{method.upper()} 识别结果:\n\n"
                for key, value in results[method].items():
                    if key != '处理时间':
                        result_text += f"{key}: {value}\n"
                
                axes[row, col].text(0.05, 0.95, result_text, 
                                  transform=axes[row, col].transAxes,
                                  fontsize=10, verticalalignment='top',
                                  bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                axes[row, col].set_title(f'{method.upper()} 结果')
                axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def performance_benchmark(self, image_paths: List[str]):
        """性能基准测试"""
        print("\n🚀 性能基准测试")
        print("=" * 60)
        
        benchmark_results = {}
        
        for method in self.methods:
            if method not in self.recognizers:
                continue
                
            print(f"\n测试 {method} 性能:")
            recognizer = self.recognizers[method]
            
            times = []
            success_count = 0
            
            for i, image_path in enumerate(image_paths):
                try:
                    start_time = time.time()
                    result = recognizer.recognize_image(image_path)
                    end_time = time.time()
                    
                    times.append(end_time - start_time)
                    success_count += 1
                    
                    print(f"  图像 {i+1}/{len(image_paths)}: {times[-1]:.2f}秒")
                    
                except Exception as e:
                    print(f"  图像 {i+1} 失败: {str(e)}")
            
            if times:
                avg_time = np.mean(times)
                min_time = np.min(times)
                max_time = np.max(times)
                
                benchmark_results[method] = {
                    'avg_time': avg_time,
                    'min_time': min_time,
                    'max_time': max_time,
                    'success_rate': (success_count / len(image_paths)) * 100
                }
                
                print(f"  平均时间: {avg_time:.2f}秒")
                print(f"  最快时间: {min_time:.2f}秒")
                print(f"  最慢时间: {max_time:.2f}秒")
                print(f"  成功率: {benchmark_results[method]['success_rate']:.1f}%")
        
        return benchmark_results
    
    def create_test_report(self, results: Dict, output_file: str):
        """生成测试报告"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': {
                'python_version': sys.version,
                'opencv_version': cv2.__version__,
            },
            'test_results': results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n📝 测试报告已保存到: {output_file}")


def create_sample_test_data():
    """创建示例测试数据"""
    sample_data = [
        {
            'image': 'sample_id1.jpg',
            'labels': {
                '姓名': '张三',
                '性别': '男',
                '公民身份号码': '110101199001010000'
            }
        },
        {
            'image': 'sample_id2.jpg', 
            'labels': {
                '姓名': '李四',
                '性别': '女',
                '公民身份号码': '110101199002020000'
            }
        }
    ]
    
    return sample_data


def main():
    print("🎭 身份证识别系统演示")
    print("=" * 50)
    
    # 初始化演示系统
    demo = OCRDemo()
    
    if not demo.recognizers:
        print("❌ 没有可用的OCR识别器，请检查安装")
        return
    
    # 检查是否有测试图像
    test_images = []
    for ext in ['.jpg', '.jpeg', '.png']:
        test_images.extend([f for f in os.listdir('.') if f.endswith(ext)])
    
    if not test_images:
        print("❌ 当前目录下没有找到测试图像")
        print("请将身份证图像放在当前目录下，支持格式: jpg, jpeg, png")
        return
    
    print(f"✓ 找到 {len(test_images)} 张测试图像")
    
    # 选择演示模式
    print("\n请选择演示模式:")
    print("1. 单图像方法对比")
    print("2. 性能基准测试")
    print("3. 批量处理演示")
    print("4. 完整演示（推荐）")
    
    choice = input("请输入选择 (1-4): ").strip()
    
    if choice == '1':
        # 单图像对比
        image_path = test_images[0]
        results, times = demo.compare_methods(image_path)
        
        # 可视化结果
        if any(results.values()):
            demo.visualize_results(image_path, results)
    
    elif choice == '2':
        # 性能测试
        benchmark_results = demo.performance_benchmark(test_images[:5])  # 限制测试图像数量
        
        # 生成报告
        demo.create_test_report(benchmark_results, 'performance_report.json')
    
    elif choice == '3':
        # 批量处理
        print(f"\n📁 批量处理 {len(test_images)} 张图像")
        
        method = 'paddleocr'  # 默认使用PaddleOCR
        if method in demo.recognizers:
            recognizer = demo.recognizers[method]
            results = recognizer.batch_recognize('.', 'batch_results.json')
            print(f"✓ 批量处理完成，结果保存到 batch_results.json")
    
    elif choice == '4':
        # 完整演示
        print("\n🎬 开始完整演示...")
        
        # 1. 方法对比
        image_path = test_images[0]
        print(f"\n第一步: 使用 {image_path} 进行方法对比")
        results, times = demo.compare_methods(image_path)
        
        # 2. 性能测试
        print(f"\n第二步: 性能基准测试")
        benchmark_results = demo.performance_benchmark(test_images[:3])
        
        # 3. 生成完整报告
        complete_report = {
            'method_comparison': results,
            'performance_benchmark': benchmark_results
        }
        demo.create_test_report(complete_report, 'complete_demo_report.json')
        
        print("\n🎉 完整演示完成！")
    
    else:
        print("❌ 无效选择")


if __name__ == "__main__":
    main() 