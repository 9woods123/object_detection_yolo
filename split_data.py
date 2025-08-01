import os
import shutil
from tqdm import tqdm

# 图片地址（根据你的实际路径设置）
image_dir = "dataset/images/"
# 标准文件地址
label_dir = "dataset/labels/"
# 训练集的比例
training_ratio = 0.9
# 拆分后数据的位置
train_dir = "train_data"

def split_data():
    # 获取所有图片文件
    images = os.listdir(image_dir)
    all = len(images)
    train_count = int(all * training_ratio)
    train_images = images[:train_count]
    val_images = images[train_count:]

    # 获取绝对路径
    image_dir_abs = os.path.abspath(image_dir)
    label_dir_abs = os.path.abspath(label_dir)
    train_dir_abs = os.path.abspath(train_dir)

    # 创建训练集和验证集目录
    os.makedirs(os.path.join(train_dir_abs, "images/train"), exist_ok=True)
    os.makedirs(os.path.join(train_dir_abs, "labels/train"), exist_ok=True)
    os.makedirs(os.path.join(train_dir_abs, "images/val"), exist_ok=True)
    os.makedirs(os.path.join(train_dir_abs, "labels/val"), exist_ok=True)

    # 写入训练集的文件列表
    with open(os.path.join(train_dir_abs, "train.txt"), "w") as file:
        file.write("\n".join([os.path.join(train_dir_abs, "images/train", img) for img in train_images]))
    print("save train.txt success!")

    # 拷贝训练集的图片和标签文件
    for img in tqdm(train_images):
        label_file = img.replace(".png", ".txt")  # 假设图片格式为 .png
        shutil.copy(os.path.join(image_dir_abs, img), os.path.join(train_dir_abs, "images/train/"))
        shutil.copy(os.path.join(label_dir_abs, label_file), os.path.join(train_dir_abs, "labels/train/"))

    # 写入验证集的文件列表
    with open(os.path.join(train_dir_abs, "val.txt"), "w") as file:
        file.write("\n".join([os.path.join(train_dir_abs, "images/val", img) for img in val_images]))
    print("save val.txt success!")

    # 拷贝验证集的图片和标签文件
    for img in tqdm(val_images):
        label_file = img.replace(".png", ".txt")  # 假设图片格式为 .png
        shutil.copy(os.path.join(image_dir_abs, img), os.path.join(train_dir_abs, "images/val/"))
        shutil.copy(os.path.join(label_dir_abs, label_file), os.path.join(train_dir_abs, "labels/val/"))

if __name__ == '__main__':
    split_data()
