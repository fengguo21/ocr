"""
调试数据集结构
"""

from datasets import load_dataset

# 加载数据集
ds = load_dataset("lansinuote/ocr_id_card")
train_ds = ds["train"]

print("=== 数据集信息 ===")
print(f"训练集大小: {len(train_ds)}")
print(f"数据集特征: {train_ds.features}")

print("\n=== 查看前几个样本 ===")
for i in range(3):
    item = train_ds[i]
    print(f"\n样本 {i}:")
    for key, value in item.items():
        if key == 'image':
            print(f"  {key}: PIL Image {value.size if hasattr(value, 'size') else type(value)}")
        elif isinstance(value, list):
            print(f"  {key}: {value[:10]}..." if len(value) > 10 else f"  {key}: {value}")
        else:
            print(f"  {key}: {value}") 