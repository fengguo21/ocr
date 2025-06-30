# 🚀 快速入门指南

这是一个5分钟的快速入门指南，帮助您快速体验身份证识别系统。

## 📋 前置条件

- Python 3.7+
- 身份证图像文件（jpg, png格式）

## ⚡ 快速安装

### 方式一：使用PaddleOCR（推荐）

```bash
# 1. 安装基础依赖
pip install torch torchvision opencv-python pillow numpy

# 2. 安装PaddleOCR
pip install paddleocr paddlepaddle

# 3. 安装其他依赖
pip install albumentations matplotlib scikit-learn tqdm
```

### 方式二：使用EasyOCR

```bash
# 1. 安装基础依赖
pip install torch torchvision opencv-python pillow numpy

# 2. 安装EasyOCR
pip install easyocr

# 3. 安装其他依赖  
pip install albumentations matplotlib scikit-learn tqdm
```

## 🎯 第一次运行

### 1. 准备测试图像
将身份证图像放在项目目录下，例如：`test_id.jpg`

### 2. 运行识别
```bash
python main.py --image test_id.jpg --method paddleocr
```

### 3. 查看结果
系统会输出类似以下的识别结果：
```
识别结果:
==================================================
姓名: 张三
性别: 男
民族: 汉族
出生: 1990年1月1日
住址: 北京市朝阳区...
公民身份号码: 110101199001010000
签发机关: 北京市公安局朝阳分局
有效期限: 2010.01.01-2030.01.01
处理时间: 2.35秒
```

## 🎮 交互式演示

运行演示脚本体验完整功能：

```bash
python demo.py
```

演示脚本提供以下功能：
- 多种OCR方法对比
- 性能基准测试
- 批量处理演示
- 可视化结果展示

## 📁 批量处理

处理整个文件夹的身份证图像：

```bash
python main.py --batch /path/to/images/ --output results.json --method paddleocr
```

## ⚙️ 自定义配置

1. 复制并修改配置文件：
```bash
cp config.json my_config.json
```

2. 编辑配置参数：
```json
{
  "ocr_method": "paddleocr",
  "preprocessing": {
    "enable_denoising": true,
    "enable_skew_correction": true
  }
}
```

3. 使用自定义配置：
```bash
python main.py --image test_id.jpg --config my_config.json
```

## 🔧 常见问题解决

### Q1: ImportError: No module named 'paddleocr'
```bash
pip install paddleocr paddlepaddle
```

### Q2: 识别结果不准确
- 确保图像清晰度足够（推荐1000px以上）
- 尝试不同的OCR方法：`--method hybrid`
- 检查图像是否包含完整的身份证信息

### Q3: 处理速度慢
- 使用GPU加速（如果可用）
- 选择更快的OCR方法：`--method paddleocr`
- 适当调整图像尺寸

### Q4: 内存不足
- 减少批处理大小
- 使用CPU模式：设置环境变量 `CUDA_VISIBLE_DEVICES=""`

## 📊 性能对比

| OCR方法 | 准确率 | 速度 | 内存使用 | 推荐场景 |
|---------|--------|------|----------|----------|
| PaddleOCR | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 生产环境 |
| EasyOCR | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 快速原型 |
| CRNN | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 资源受限 |
| Hybrid | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | 最高精度 |

## 🎯 下一步

- 📖 阅读完整的 [README.md](README.md) 了解更多功能
- 🧪 运行 `python demo.py` 体验所有功能
- 🎓 查看 [训练自定义模型](README.md#训练自定义模型) 章节
- 🛠️ 探索 [API使用](README.md#api-使用) 进行二次开发

## 💡 小贴士

1. **首次运行**会下载预训练模型，请确保网络连接正常
2. **GPU加速**可显著提升处理速度
3. **图像质量**是影响识别准确率的关键因素
4. **混合方案**能获得最高精度，但处理时间较长

---

🎉 恭喜！您已经成功运行了身份证识别系统。如有问题，请查看 [README.md](README.md) 或提交 Issue。 