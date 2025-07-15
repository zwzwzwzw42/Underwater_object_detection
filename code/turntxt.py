mport xml.etree.ElementTree as ET
import os

# 定义类别及其对应的编号
class_mapping = {
    'holothurian': 0,
    'echinus': 1,
    'scallop': 2,
    'starfish': 3
    # 'waterweeds' 将被忽略
}

def convert_xml_to_txt(xml_path, txt_path):
    try:
        # 解析XML文件（使用原始字符串处理路径）
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 获取图像尺寸
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        
        # 准备写入TXT文件
        with open(txt_path, 'w') as f:
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                
                # 忽略水草类别
                if class_name == 'waterweeds':
                    continue
                    
                # 检查类别是否在映射中
                if class_name not in class_mapping:
                    print(f"警告: 未知类别 '{class_name}'，将被忽略")
                    continue
                    
                # 获取边界框坐标
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                
                # 计算中心点和宽高
                x_center = (xmin + xmax) / 2 / img_width
                y_center = (ymin + ymax) / 2 / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                
                # 确保坐标在0-1范围内
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                # 写入TXT文件
                f.write(f"{class_mapping[class_name]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    except Exception as e:
        print(f"处理文件 {xml_path} 时出错: {str(e)}")
        raise

# 批量处理函数
def batch_convert_xml_to_txt(xml_dir, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历XML目录
    for xml_file in os.listdir(xml_dir):
        if xml_file.lower().endswith('.xml'):
            # 正确处理路径（使用原始字符串或双反斜杠）
            xml_path = os.path.join(xml_dir, xml_file)
            txt_file = os.path.splitext(xml_file)[0] + ".txt"
            txt_path = os.path.join(output_dir, txt_file)
            
            try:
                convert_xml_to_txt(xml_path, txt_path)
                print(f"成功转换: {xml_path} -> {txt_path}")
            except Exception as e:
                print(f"转换失败: {xml_path} - {str(e)}")

# 示例用法（使用原始字符串或双反斜杠）
xml_directory = r"E:\BaiduNetdiskDownload\train\box"  # 原始字符串
# 或者
# xml_directory = "E:\\BaiduNetdiskDownload\\train\\box"  # 双反斜杠

output_directory = r"E:\BaiduNetdiskDownload\train\labels"  # 输出目录

batch_convert_xml_to_txt(xml_directory, output_directory)