"""
身份证识别系统主程序
支持多种OCR方案：
1. 自定义CRNN模型
2. PaddleOCR
3. EasyOCR
4. 混合方案
"""

import os
import sys
import cv2
import numpy as np
import argparse
from typing import Dict, List, Tuple, Optional
import json
import time

# 添加模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 尝试导入深度学习依赖
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch未安装，CRNN模型功能将不可用")

from utils.image_processing import IDCardPreprocessor
from utils.id_info_extractor import IDCardInfoExtractor

# 只有在PyTorch可用时才导入模型
if TORCH_AVAILABLE:
    from models.crnn import CRNN, AttentionCRNN
    from models.text_detection import TextDetector


class IDCardRecognizer:
    """身份证识别器主类"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 初始化组件
        self.preprocessor = IDCardPreprocessor()
        self.info_extractor = IDCardInfoExtractor()
        
        # 初始化OCR模型
        self.ocr_method = config.get('ocr_method', 'paddleocr')
        
        # 检查深度学习支持
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'
        
        self._init_ocr_models()
        
        print(f"初始化完成，使用设备: {self.device}")
        print(f"OCR方法: {self.ocr_method}")

    def _init_ocr_models(self):
        """初始化OCR模型"""
        if self.ocr_method == 'paddleocr':
            self._init_paddleocr()
        elif self.ocr_method == 'easyocr':
            self._init_easyocr()
        elif self.ocr_method == 'crnn':
            if TORCH_AVAILABLE:
                self._init_crnn()
            else:
                print("❌ CRNN模型需要PyTorch支持，请安装PyTorch或使用其他OCR方法")
                sys.exit(1)
        elif self.ocr_method == 'hybrid':
            self._init_hybrid()
        else:
            raise ValueError(f"不支持的OCR方法: {self.ocr_method}")

    def _init_paddleocr(self):
        """初始化PaddleOCR"""
        try:
            from paddleocr import PaddleOCR
            # 使用最简单的参数配置
            self.ocr_engine = PaddleOCR(lang='ch')
            print("PaddleOCR初始化成功")
        except ImportError:
            print("PaddleOCR未安装，请运行: pip install paddleocr")
            sys.exit(1)
        except Exception as e:
            print(f"PaddleOCR初始化失败: {e}")
            sys.exit(1)

    def _init_easyocr(self):
        """初始化EasyOCR"""
        try:
            import easyocr
            gpu_available = TORCH_AVAILABLE and torch.cuda.is_available()
            self.ocr_engine = easyocr.Reader(['ch_sim', 'en'], gpu=gpu_available)
            print("EasyOCR初始化成功")
        except ImportError:
            print("EasyOCR未安装，请运行: pip install easyocr")
            sys.exit(1)

    def _init_crnn(self):
        """初始化自定义CRNN模型"""
        if not TORCH_AVAILABLE:
            raise ImportError("CRNN模型需要PyTorch支持")
            
        # 字符集定义
        self.charset = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' + \
                      '中华人民共和国身份证姓名性别民族出生年月日住址公民号码签发机关有效期限长至汉族男女' + \
                      '省市县区街道路号楼室派出所公安局厅'
        
        self.char_to_idx = {char: idx for idx, char in enumerate(self.charset)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.charset)}
        
        # 初始化模型
        model_config = self.config.get('crnn_config', {})
        self.crnn_model = CRNN(
            img_h=model_config.get('img_h', 32),
            nc=model_config.get('nc', 1),
            nclass=len(self.charset) + 1,  # +1 for blank
            nh=model_config.get('nh', 256)
        ).to(self.device)
        
        # 加载预训练模型（如果存在）
        model_path = model_config.get('model_path')
        if model_path and os.path.exists(model_path):
            self.crnn_model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"加载CRNN模型: {model_path}")
        else:
            print("未找到预训练CRNN模型，将使用随机初始化的模型")
        
        self.crnn_model.eval()
        
        # 初始化文本检测器
        self.text_detector = TextDetector(device=self.device)

    def _init_hybrid(self):
        """初始化混合方案"""
        self._init_paddleocr()
        self._init_easyocr()
        print("混合OCR方案初始化成功")

    def recognize_image(self, image_path: str) -> Dict[str, str]:
        """
        识别身份证图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            识别结果字典
        """
        start_time = time.time()
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        print(f"开始处理图像: {image_path}")
        
        # 图像预处理
        processed_image = self.preprocessor.preprocess_for_ocr(image, training=False)
        print(f"预处理完成，图像尺寸: {processed_image.shape}")
        
        # OCR识别
        ocr_results = self._perform_ocr(processed_image)
        print(f"OCR识别到{len(ocr_results)}个文本区域")
        
        # 显示识别的文本内容（用于调试）
        if ocr_results:
            print("识别的文本内容:")
            for i, (text, confidence) in enumerate(ocr_results):
                print(f"  {i+1}. '{text}' (置信度: {confidence:.3f})")
        else:
            print("⚠️  未识别到任何文本")
        
        # 信息提取
        # 首先判断身份证正面还是背面
        full_text = ' '.join([text for text, _ in ocr_results])
        is_front = self.info_extractor._is_front_side(full_text, ocr_results)
        print(f"身份证面别判断: {'正面' if is_front else '背面'}")
        
        id_info = self.info_extractor.extract_info(ocr_results)
        
        # 计算耗时
        processing_time = time.time() - start_time
        id_info['处理时间'] = f"{processing_time:.2f}秒"
        
        print(f"识别完成，耗时: {processing_time:.2f}秒")
        
        return id_info

    def _perform_ocr(self, image: np.ndarray) -> List[Tuple[str, float]]:
        """执行OCR识别"""
        if self.ocr_method == 'paddleocr':
            return self._paddleocr_recognize(image)
        elif self.ocr_method == 'easyocr':
            return self._easyocr_recognize(image)
        elif self.ocr_method == 'crnn':
            return self._crnn_recognize(image)
        elif self.ocr_method == 'hybrid':
            return self._hybrid_recognize(image)
        else:
            raise ValueError(f"不支持的OCR方法: {self.ocr_method}")

    def _paddleocr_recognize(self, image: np.ndarray) -> List[Tuple[str, float]]:
        """PaddleOCR识别"""
        try:
            # 使用新版PaddleOCR的predict方法
            if hasattr(self.ocr_engine, 'predict'):
                results = self.ocr_engine.predict(image)
                print(f"PaddleOCR predict结果类型: {type(results)}")
                print(f"PaddleOCR predict结果: {results}")
            else:
                # 尝试旧版本的ocr方法
                results = self.ocr_engine.ocr(image)
                
        except Exception as e:
            print(f"PaddleOCR调用失败: {e}")
            return []
        
        ocr_results = []
        
        # 处理新版本PaddleOCR的结果格式
        try:
            if results and isinstance(results, list) and len(results) > 0:
                # 获取第一个结果项
                result_item = results[0]
                
                if isinstance(result_item, dict):
                    # 新版PaddleOCR格式：包含rec_texts和rec_scores
                    if 'rec_texts' in result_item and 'rec_scores' in result_item:
                        texts = result_item.get('rec_texts', [])
                        scores = result_item.get('rec_scores', [])
                        
                        print(f"识别到的文本: {texts}")
                        print(f"对应的置信度: {scores}")
                        
                        for text, score in zip(texts, scores):
                            if text and text.strip():  # 过滤空文本
                                ocr_results.append((text.strip(), float(score)))
                    
                    # 其他可能的格式
                    elif 'rec_text' in result_item:
                        text = result_item.get('rec_text', '')
                        score = result_item.get('rec_score', 0.0)
                        if text.strip():
                            ocr_results.append((text, float(score)))
                    elif 'text' in result_item:
                        text = result_item.get('text', '')
                        score = result_item.get('score', 0.0)
                        if text.strip():
                            ocr_results.append((text, float(score)))
                
        except Exception as e:
            print(f"解析PaddleOCR结果失败: {e}")
            print(f"原始结果类型: {type(results)}")
            if results:
                print(f"第一个结果项类型: {type(results[0])}")
                if isinstance(results[0], dict):
                    print(f"字典键: {list(results[0].keys())}")
        
        return ocr_results

    def _easyocr_recognize(self, image: np.ndarray) -> List[Tuple[str, float]]:
        """EasyOCR识别"""
        results = self.ocr_engine.readtext(image)
        ocr_results = []
        
        for item in results:
            text = item[1]
            confidence = item[2]
            ocr_results.append((text, confidence))
        
        return ocr_results

    def _crnn_recognize(self, image: np.ndarray) -> List[Tuple[str, float]]:
        """CRNN识别"""
        if not TORCH_AVAILABLE:
            raise ImportError("CRNN识别需要PyTorch支持")
        
        # 确保图像是3通道的（TextDetector需要RGB输入）
        if len(image.shape) == 2:
            # 如果是灰度图，转换为3通道
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 1:
            # 如果是单通道，复制到3通道
            image = np.repeat(image, 3, axis=2)
            
        try:
            # 文本检测
            boxes, _ = self.text_detector.detect_text(image)
            print(f"文本检测结果: {boxes}")
            
            ocr_results = []
            for box in boxes:
                # 提取文本区域
                x1, y1 = int(box[0][0]), int(box[0][1])
                x2, y2 = int(box[2][0]), int(box[2][1])
                
                # 边界检查
                h, w = image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:  # 确保区域有效
                    text_region = image[y1:y2, x1:x2]
                    
                    # 文本识别
                    text, confidence = self._recognize_text_region(text_region)
                    if text.strip():
                        ocr_results.append((text, confidence))
            
            return ocr_results
            
        except Exception as e:
            print(f"CRNN文本检测失败: {e}")
            # 如果文本检测失败，直接对整个图像进行识别
            return self._fallback_crnn_recognition(image)

    def _recognize_text_region(self, region: np.ndarray) -> Tuple[str, float]:
        """识别单个文本区域"""
        # 预处理文本区域
        region = self.preprocessor.preprocess_text_region(region)
        
        # 调整尺寸
        h, w = region.shape
        target_h = 32
        target_w = int(w * target_h / h)
        region = cv2.resize(region, (target_w, target_h))
        
        # 转换为tensor
        region = region.astype(np.float32) / 255.0
        region = torch.from_numpy(region).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # 模型推理
        with torch.no_grad():
            output = self.crnn_model(region)
            output = torch.nn.functional.log_softmax(output, dim=2)
        
        # 解码结果
        text, confidence = self._decode_crnn_output(output)
        
        return text, confidence

    def _decode_crnn_output(self, output: 'torch.Tensor') -> Tuple[str, float]:
        """解码CRNN输出"""
        # 简单的贪心解码
        output = output.squeeze(0).cpu().numpy()
        
        # 获取最可能的字符序列
        predicted_chars = []
        confidences = []
        
        for t in range(output.shape[0]):
            char_probs = output[t]
            char_idx = np.argmax(char_probs)
            
            if char_idx != len(self.charset):  # 不是blank
                if char_idx in self.idx_to_char:
                    predicted_chars.append(self.idx_to_char[char_idx])
                    confidences.append(np.exp(char_probs[char_idx]))
        
        # 去除重复字符
        if predicted_chars:
            final_text = predicted_chars[0]
            final_confidences = [confidences[0]]
            
            for i in range(1, len(predicted_chars)):
                if predicted_chars[i] != predicted_chars[i-1]:
                    final_text += predicted_chars[i]
                    final_confidences.append(confidences[i])
            
            avg_confidence = np.mean(final_confidences) if final_confidences else 0.0
            return final_text, avg_confidence
        
        return "", 0.0

    def _fallback_crnn_recognition(self, image: np.ndarray) -> List[Tuple[str, float]]:
        """
        当文本检测失败时的fallback识别方法
        直接将整个图像分割成可能的文本区域进行识别
        """
        try:
            # 使用传统的文本区域提取方法
            text_regions = self.preprocessor.extract_text_regions(image)
            
            ocr_results = []
            for region in text_regions:
                text, confidence = self._recognize_text_region(region)
                if text.strip():
                    ocr_results.append((text, confidence))
            
            # 如果传统方法也没有结果，直接对整个图像进行识别
            if not ocr_results:
                print("使用整个图像进行CRNN识别...")
                text, confidence = self._recognize_text_region(image)
                if text.strip():
                    ocr_results.append((text, confidence))
            
            return ocr_results
            
        except Exception as e:
            print(f"Fallback识别也失败: {e}")
            return []

    def _hybrid_recognize(self, image: np.ndarray) -> List[Tuple[str, float]]:
        """混合识别方案"""
        # 使用多个OCR引擎识别
        paddle_results = []
        easy_results = []
        
        # 安全调用各个OCR引擎
        try:
            paddle_results = self._paddleocr_recognize(image)
        except Exception as e:
            print(f"PaddleOCR识别失败: {e}")
        
        try:
            easy_results = self._easyocr_recognize(image)
        except Exception as e:
            print(f"EasyOCR识别失败: {e}")
        
        # 合并结果（简单策略：选择置信度较高的结果）
        combined_results = []
        
        # 将所有结果放入字典，按文本内容分组
        text_dict = {}
        
        for text, conf in paddle_results:
            if text not in text_dict or conf > text_dict[text]:
                text_dict[text] = conf
        
        for text, conf in easy_results:
            if text not in text_dict or conf > text_dict[text]:
                text_dict[text] = conf
        
        # 转换回列表格式
        combined_results = [(text, conf) for text, conf in text_dict.items()]
        
        return combined_results

    def batch_recognize(self, image_dir: str, output_file: str = None):
        """批量识别"""
        results = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        for filename in os.listdir(image_dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(image_dir, filename)
                try:
                    result = self.recognize_image(image_path)
                    result['文件名'] = filename
                    results.append(result)
                    print(f"完成: {filename}")
                except Exception as e:
                    print(f"处理 {filename} 时出错: {str(e)}")
        
        # 保存结果
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"结果已保存到: {output_file}")
        
        return results


def check_dependencies():
    """检查依赖环境"""
    print("🔍 检查系统依赖...")
    
    dependencies = {
        'OpenCV': cv2.__version__,
        'NumPy': np.__version__,
    }
    
    if TORCH_AVAILABLE:
        dependencies['PyTorch'] = torch.__version__
        dependencies['CUDA支持'] = '是' if torch.cuda.is_available() else '否'
    else:
        dependencies['PyTorch'] = '未安装'
    
    # 检查OCR引擎
    try:
        import paddleocr
        dependencies['PaddleOCR'] = '已安装'
    except ImportError:
        dependencies['PaddleOCR'] = '未安装'
    
    try:
        import easyocr
        dependencies['EasyOCR'] = '已安装'
    except ImportError:
        dependencies['EasyOCR'] = '未安装'
    
    print("\n📋 依赖检查结果:")
    for name, version in dependencies.items():
        print(f"  {name}: {version}")
    
    # 推荐OCR方法
    if dependencies.get('PaddleOCR') == '已安装':
        print("\n✅ 推荐使用 PaddleOCR (--method paddleocr)")
    elif dependencies.get('EasyOCR') == '已安装':
        print("\n✅ 推荐使用 EasyOCR (--method easyocr)")
    else:
        print("\n⚠️  建议安装 PaddleOCR 或 EasyOCR")
        print("   pip install paddleocr paddlepaddle")
        print("   pip install easyocr")


def main():
    parser = argparse.ArgumentParser(description='身份证识别系统')
    parser.add_argument('--image', type=str, help='输入图像路径')
    parser.add_argument('--batch', type=str, help='批量处理目录')
    parser.add_argument('--output', type=str, help='输出文件路径')
    parser.add_argument('--method', type=str, default='paddleocr', 
                       choices=['paddleocr', 'easyocr', 'crnn', 'hybrid'],
                       help='OCR方法')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--check-deps', action='store_true', help='检查依赖环境')
    
    args = parser.parse_args()
    
    # 检查依赖
    if args.check_deps:
        check_dependencies()
        return
    
    # 如果没有任何参数，显示依赖检查
    if not any([args.image, args.batch]):
        check_dependencies()
        print("\n💡 使用示例:")
        print("  python main.py --image idcard.jpg --method paddleocr")
        print("  python main.py --batch images/ --output results.json")
        print("  python main.py --check-deps")
        return
    
    # 加载配置
    config = {'ocr_method': args.method}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config.update(json.load(f))
    
    # 初始化识别器
    try:
        recognizer = IDCardRecognizer(config)
    except Exception as e:
        print(f"❌ 初始化失败: {str(e)}")
        print("请检查依赖安装或使用 --check-deps 检查环境")
        return
    
    if args.image:
        # 单张图像识别
        try:
            result = recognizer.recognize_image(args.image)
            print("\n识别结果:")
            print("=" * 50)
            for key, value in result.items():
                print(f"{key}: {value}")
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"\n结果已保存到: {args.output}")
        except Exception as e:
            print(f"❌ 识别失败: {str(e)}")
    
    elif args.batch:
        # 批量识别
        try:
            results = recognizer.batch_recognize(args.batch, args.output)
            print(f"\n批量处理完成，共处理 {len(results)} 张图像")
        except Exception as e:
            print(f"❌ 批量处理失败: {str(e)}")


if __name__ == "__main__":
    # 硬编码参数 - 直接运行时使用这些参数
    import sys
    sys.argv = ['main.py', '--image', 'id.jpeg', '--method', 'crnn']
    
    main()
