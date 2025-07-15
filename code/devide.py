import os
import shutil
import random
from sklearn.model_selection import train_test_split

# 设置随机种子以确保可重复性
random.seed(42)

# 原始路径设置
image_dir = r"E:\BaiduNetdiskDownload\train\image"  # 原始图像目录
label_dir = r"E:\BaiduNetdiskDownload\train\labels"  # 标签文件目录
output_dir = r"E:\BaiduNetdiskDownload\train1"  # 输出根目录

# 类别定义（必须保持这个顺序）
class_mapping = {
    'holothurian': 0,
    'echinus': 1,
    'scallop': 2,
    'starfish': 3
}

# 创建输出目录结构
train_image_dir = os.path.join(output_dir, "train", "images")
train_label_dir = os.path.join(output_dir, "train", "labels")
val_image_dir = os.path.join(output_dir, "val", "images")
val_label_dir = os.path.join(output_dir, "val", "labels")

os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# 获取所有图像文件名（不带扩展名）
image_files = [f.split('.')[0] for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"找到 {len(image_files)} 个图像文件")

# 检查对应的标签文件是否存在
valid_files = []
for file in image_files:
    label_file = os.path.join(label_dir, f"{file}.txt")
    if os.path.exists(label_file):
        valid_files.append(file)
    else:
        print(f"警告: 图像 {file} 没有对应的标签文件")

print(f"有效图像-标签对: {len(valid_files)} 个")

# 按8:2比例划分训练集和验证集
train_files, val_files = train_test_split(valid_files, test_size=0.2, random_state=42)

# 复制训练集文件
print("\n正在复制训练集文件...")
for file in train_files:
    # 复制图像文件
    src_image = os.path.join(image_dir, f"{file}.jpg")
    dst_image = os.path.join(train_image_dir, f"{file}.jpg")
    shutil.copy2(src_image, dst_image)
    
    # 复制标签文件
    src_label = os.path.join(label_dir, f"{file}.txt")
    dst_label = os.path.join(train_label_dir, f"{file}.txt")
    shutil.copy2(src_label, dst_label)

# 复制验证集文件
print("\n正在复制验证集文件...")
for file in val_files:
    # 复制图像文件
    src_image = os.path.join(image_dir, f"{file}.jpg")
    dst_image = os.path.join(val_image_dir, f"{file}.jpg")
    shutil.copy2(src_image, dst_image)
    
    # 复制标签文件
    src_label = os.path.join(label_dir, f"{file}.txt")
    dst_label = os.path.join(val_label_dir, f"{file}.txt")
    shutil.copy2(src_label, dst_label)

# 创建YOLOv8所需的dataset.yaml文件（确保类别顺序与class_mapping一致）
yaml_content = f"""path: {output_dir}
train: train/images
val: val/images

# 类别数量和名称（顺序必须与class_mapping一致）
nc: {len(class_mapping)}
names: {list(class_mapping.keys())}
"""

yaml_path = os.path.join(output_dir, "dataset.yaml")
with open(yaml_path, 'w') as f:
    f.write(yaml_content)

print(f"\n数据集划分完成！")
print(f"训练集大小: {len(train_files)}")
print(f"验证集大小: {len(val_files)}")
print(f"类别映射: {class_mapping}")
print(f"数据集配置文件已保存到: {yaml_path}")
print(f"最终目录结构:")
print(f"{output_dir}")
print(f"├── train/")
print(f"│   ├── images/")
print(f"│   └── labels/")
print(f"├── val/")
print(f"│   ├── images/")
print(f"│   └── labels/")
print(f"└── dataset.yaml")