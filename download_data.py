from ultralytics.utils.downloads import download
from pathlib import Path
import zipfile

def download_and_prepare_coco(base_path: str, segments: bool = True):
    """
    下载并解压 COCO 2017 数据集和标注文件。

    Args:
        base_path (str): 数据集存储的根目录。
        segments (bool): 是否下载分割标注，默认为 True；否则下载边界框标注。
    """
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    # 下载标注文件
    url = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/'
    labels_url = url + ('coco2017labels-segments.zip' if segments else 'coco2017labels.zip')
    print(f"Downloading labels from {labels_url}...")
    download([labels_url], dir=base_path)

    # 下载图片数据集
    image_urls = [
        'http://images.cocodataset.org/zips/train2017.zip',  # 训练集
        'http://images.cocodataset.org/zips/val2017.zip',    # 验证集
        'http://images.cocodataset.org/zips/test2017.zip'    # 测试集（可选）
    ]
    print("Downloading images...")
    download(image_urls, dir=base_path / 'images', threads=3)

    # 解压文件
    print("Extracting downloaded files...")
    for zip_file in base_path.rglob('*.zip'):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(zip_file.parent)
        zip_file.unlink()  # 删除压缩包

    print("COCO 2017 dataset and labels are ready!")

# 使用示例
if __name__ == "__main__":
    # 设置数据集存储路径
    dataset_path = "coco_dataset" 
    # # 下载 COCO 数据集，选择分割标注 (segments=True) 或边界框标注 (segments=False)
    download_and_prepare_coco(dataset_path, segments=True)
