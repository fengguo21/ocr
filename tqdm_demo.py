"""
演示 tqdm 工作原理的示例代码
对比 print 和 tqdm 的不同行为
"""

import sys
import time
from tqdm import tqdm


def demo_print():
    """使用普通 print 的效果"""
    print("=== 使用普通 print ===")
    for i in range(5):
        print(f"进度: {i}/5")
        time.sleep(0.5)
    print("完成！\n")


def demo_manual_progress():
    """手动实现进度条（模拟 tqdm 原理）"""
    print("=== 手动实现进度条 ===")
    total = 10
    
    for i in range(total + 1):
        # 计算进度
        percent = (i / total) * 100
        bar_length = 20
        filled = int(bar_length * i / total)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        # \r 回到行首，覆盖之前的内容
        progress = f"\r进度: |{bar}| {percent:5.1f}% ({i}/{total})"
        sys.stdout.write(progress)
        sys.stdout.flush()  # 立即显示
        
        time.sleep(0.3)
    
    print("\n完成！\n")  # 换行


def demo_tqdm():
    """使用 tqdm 的效果"""
    print("=== 使用 tqdm ===")
    
    # 基础用法
    for i in tqdm(range(10), desc="基础进度"):
        time.sleep(0.2)
    
    # 带自定义信息
    pbar = tqdm(range(10), desc="训练中")
    for i in pbar:
        loss = 1.0 / (i + 1)  # 模拟损失下降
        pbar.set_postfix({"loss": f"{loss:.4f}"})
        time.sleep(0.2)
    
    print("完成！\n")


def show_control_characters():
    """展示控制字符的作用"""
    print("=== 控制字符演示 ===")
    
    print("正常输出：Hello World")
    time.sleep(1)
    
    print("使用 \\r 回到行首：", end="")
    sys.stdout.flush()
    time.sleep(1)
    print("\rNew Content")  # 会覆盖之前的内容
    time.sleep(1)
    
    print("清除行内容：", end="")
    sys.stdout.flush()
    time.sleep(1)
    print("\r\033[K新的内容")  # \033[K 清除行内容
    time.sleep(1)


if __name__ == "__main__":
    print("🔍 tqdm 原理演示\n")
    
    # 1. 对比 print 和手动进度条
    demo_print()
    demo_manual_progress()
    
    # 2. tqdm 的实际效果
    demo_tqdm()
    
    # 3. 底层控制字符
    show_control_characters()
    
    print("\n✅ 演示完成！")
    print("\n📝 总结：")
    print("- print: 每次输出新行")
    print("- tqdm: 使用 \\r 和 ANSI 转义序列在同一行更新")
    print("- 核心：sys.stdout.write() + flush() + 控制字符") 