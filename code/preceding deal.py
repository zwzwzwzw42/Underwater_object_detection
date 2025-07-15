import os
import cv2
import numpy as np
from tqdm import tqdm

# 硬编码路径配置
INPUT_VIDEO = r"E:\BaiduNetdiskDownload\YN020013.MP4"
OUTPUT_DIR = r"E:\BaiduNetdiskDownload\train1\deal"

def process_video(input_path, output_dir, frame_interval=10, target_size=(640, 640), 
                 skip_seconds=0, max_frames=None):
    """
    处理视频文件的专用函数
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取视频信息
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {input_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"\n正在处理视频: {os.path.basename(input_path)}")
    print(f"视频时长: {duration:.2f}秒, 帧率: {fps:.2f}, 总帧数: {total_frames}")
    print(f"输出尺寸: {target_size}, 帧间隔: {frame_interval}")

    # 跳过开头
    skip_frames = int(skip_seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
    
    # 准备处理
    frame_count = 0
    saved_count = 0
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # 进度条设置
    pbar = tqdm(total=min(max_frames or total_frames, total_frames - skip_frames), 
                desc="提取进度", unit="帧")

    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and saved_count >= max_frames):
            break
            
        if frame_count % frame_interval == 0:
            # 智能缩放和填充
            h, w = frame.shape[:2]
            scale = min(target_size[0] / w, target_size[1] / h)
            resized = cv2.resize(frame, None, fx=scale, fy=scale)
            
            # 创建填充图像
            padded = np.full((target_size[1], target_size[0], 3), 114, dtype=np.uint8)
            pad_h = (target_size[1] - resized.shape[0]) // 2
            pad_w = (target_size[0] - resized.shape[1]) // 2
            padded[pad_h:pad_h+resized.shape[0], pad_w:pad_w+resized.shape[1]] = resized
            
            # 保存图像
            output_path = os.path.join(output_dir, f"{base_name}_frame{frame_count+skip_frames:06d}.jpg")
            cv2.imwrite(output_path, padded, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1
            
        frame_count += 1
        pbar.update(1)
    
    cap.release()
    pbar.close()
    
    print(f"\n处理完成！共保存 {saved_count} 帧图像到: {output_dir}")

if __name__ == "__main__":
    # 直接调用处理函数
    process_video(
        input_path=INPUT_VIDEO,
        output_dir=OUTPUT_DIR,
        frame_interval=10,       # 每10帧提取1帧
        target_size=(640, 640),  # 输出640x640
        skip_seconds=0,          # 不跳过开头
        max_frames=None          # 不限制最大帧数
    )
