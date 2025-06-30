"""
使用训练好的CRAFT模型进行文本检测的简单示例
"""

import cv2
import matplotlib.pyplot as plt
from models.text_detection import TextDetector


def detect_with_trained_model(image_path, model_path=None):
    """
    使用训练好的CRAFT模型检测文本
    
    Args:
        image_path: 输入图像路径
        model_path: 训练好的模型权重路径（可选）
    """
    
    # 创建文本检测器
    print("🚀 初始化CRAFT文本检测器...")
    detector = TextDetector(
        model_path=model_path,  # 训练好的权重路径
        device='cpu',           # 可改为'cuda'
        use_pretrained=True     # 如果没有自定义权重，使用预训练权重
    )
    
    # 进行文本检测
    print(f"📸 检测图像: {image_path}")
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图像: {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 检测文本
    boxes, polys = detector.detect_text(
        image_rgb,
        text_threshold=0.7,    # 文本检测阈值
        link_threshold=0.4,    # 链接检测阈值
        low_text=0.4          # 低文本阈值
    )
    
    print(f"🎯 检测结果: 找到 {len(boxes)} 个文本区域")
    
    # 可视化结果
    plt.figure(figsize=(12, 8))
    
    # 在图像上绘制检测框
    result_img = image_rgb.copy()
    for i, box in enumerate(boxes):
        if box is not None:
            # 转换为整数坐标
            box = box.astype(int)
            # 绘制多边形
            pts = box.reshape((-1, 1, 2))
            cv2.polylines(result_img, [pts], True, (0, 255, 0), 2)
            # 添加序号
            cv2.putText(result_img, str(i+1), tuple(box[0]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # 显示结果
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title('原图')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(result_img)
    plt.title(f'检测结果 ({len(boxes)} 个文本区域)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('detection_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ 检测完成！结果已保存为 'detection_result.png'")
    
    return boxes, polys


if __name__ == "__main__":
    # 使用示例
    
    # 1. 使用预训练权重（无自定义训练）
    print("=== 示例1: 使用预训练权重 ===")
    detect_with_trained_model(
        image_path="id.jpeg",  # 替换为您的图像路径
        model_path=None        # 不使用自定义权重
    )
    
    # 2. 使用训练好的权重（如果有的话）
    print("\n=== 示例2: 使用训练好的权重 ===")
    trained_model_path = "checkpoints_custom/best_model.pth"  # 替换为您的模型路径
    
    detect_with_trained_model(
        image_path="id.jpeg",           # 替换为您的图像路径
        model_path=trained_model_path   # 使用训练好的权重
    )
    
    print("\n📚 使用说明:")
    print("1. 如果您已经训练了模型，将 model_path 设置为您的 .pth 文件路径")
    print("2. 如果没有训练过，设置 model_path=None 使用预训练权重")
    print("3. 可以通过 python train_with_your_data.py 训练自己的模型")
    print("4. 训练好的模型会保存在 checkpoints_custom/ 目录下") 