"""
身份证识别系统安装脚本
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="idcard-ocr",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="基于PyTorch的身份证信息识别系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/idcard-ocr",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.812",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",
            "torchvision>=0.15.0+cu118",
        ],
    },
    entry_points={
        "console_scripts": [
            "idcard-ocr=main:main",
            "idcard-train=train_crnn:main",
            "idcard-demo=demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.md", "*.txt"],
    },
    keywords="ocr, pytorch, idcard, recognition, computer-vision",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/idcard-ocr/issues",
        "Source": "https://github.com/yourusername/idcard-ocr",
    },
) 