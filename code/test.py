import cv2
from ultralytics import YOLO
import time

# 加载训练好的模型
model_path = r"C:\Users\DELL\Desktop\underwater_detection\exp1\weights\best.pt"
model = YOLO(model_path)

# 视频文件路径
video_path = r"E:\BaiduNetdiskDownload\YN020013.MP4"  # 替换为您的视频路径

# 初始化视频捕获
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"视频原始帧率: {fps:.2f} FPS")

# 性能统计变量
frame_count = 0
total_time = 0
fps_list = []
loss_list = []

# 检测循环
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 记录开始时间
    start_time = time.time()
    
    # 执行检测
    results = model(frame, verbose=False)  # 禁用详细日志
    
    # 计算处理时间
    process_time = time.time() - start_time
    current_fps = 1 / process_time
    
    # 收集性能数据
    fps_list.append(current_fps)
    if results[0].speed['inference'] > 0:  # 确保有检测结果
        loss_list.append(results[0].boxes.conf.mean().item())  # 使用检测置信度作为loss代理
    
    # 打印实时信息
    print(f"Frame {frame_count} | FPS: {current_fps:.2f} | Avg Confidence: {results[0].boxes.conf.mean().item():.4f}")
    
    # 可视化结果
    annotated_frame = results[0].plot()  # 自动绘制检测框
    cv2.imshow("Underwater Detection", annotated_frame)
    
    # 退出条件
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_count += 1
    total_time += process_time

# 释放资源
cap.release()
cv2.destroyAllWindows()

# 性能统计报告
avg_fps = frame_count / total_time
avg_loss = sum(loss_list) / len(loss_list) if loss_list else 0

print("\n===== 检测报告 =====")
print(f"总帧数: {frame_count}")
print(f"平均帧率: {avg_fps:.2f} FPS")
print(f"平均检测置信度: {avg_loss:.4f}")
print(f"最高单帧FPS: {max(fps_list):.2f}")
print(f"最低单帧FPS: {min(fps_list):.2f}")