import os
import xml.etree.ElementTree as ET
from PIL import Image

# 创建类别字典
# 假设你有一个类别字典，其中 0 是 'car' 类，1 是 'person' 类，依此类推
class_dict = {'person': 0, 'tank': 1}

def convert_xml_to_yolo(xml_path, img_width, img_height):
    """
    将 Pascal VOC XML 格式转换为 YOLO 格式。
    
    参数：
        xml_path (str): XML 文件的路径。
        img_width (int): 图像的宽度。
        img_height (int): 图像的高度。
        
    返回：
        list: 转换后的 YOLO 格式标注，格式为 [class_id, x_center, y_center, width, height]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    yolo_annotations = []
    
    # 遍历所有目标
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        class_id = class_dict.get(class_name)
        
        # 获取边界框坐标
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
        # 计算 YOLO 格式的归一化坐标
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        # 保存转换后的标注
        yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
    
    return yolo_annotations

def convert_dataset(xml_dir, img_dir, output_dir):
    """
    将一个文件夹中的所有 XML 标注转换为 YOLO 格式，并保存在 output_dir 中。
    
    参数：
        xml_dir (str): XML 文件所在的文件夹路径。
        img_dir (str): 图像文件所在的文件夹路径。
        output_dir (str): 输出的 YOLO 格式标注文件夹路径。
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历 XML 文件并进行转换
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(xml_dir, xml_file)
            
            # 获取图像尺寸（假设图像文件名和 XML 文件名相同）
            img_file = xml_file.replace('.xml', '.png')  # 假设图像是 .jpg 格式
            img_path = os.path.join(img_dir, img_file)
            
            # 获取图像尺寸
            img = Image.open(img_path)
            img_width, img_height = img.size
            
            # 转换为 YOLO 格式
            yolo_annotations = convert_xml_to_yolo(xml_path, img_width, img_height)
            
            # 保存为 .txt 文件
            txt_file = os.path.join(output_dir, xml_file.replace('.xml', '.txt'))
            with open(txt_file, 'w') as f:
                f.write("\n".join(yolo_annotations))

def main():
    """
    主函数，用于执行数据转换工作流。
    """
    # 用户指定路径
    xml_dir = 'xml'  # XML 文件所在目录
    img_dir = 'images'           # 图像文件所在目录
    output_dir = 'labels'        # 转换后的 YOLO 格式标注文件存储目录
    
    # 调用转换函数
    convert_dataset(xml_dir, img_dir, output_dir)
    print(f"YOLO 格式标注文件已保存到: {output_dir}")

if __name__ == "__main__":
    main()
