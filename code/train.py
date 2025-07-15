from ultralytics import YOLO
import torch
import os

# 检查GPU可用性
print("="*50)
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"当前GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# 配置数据集路径
dataset_yaml = r"E:\BaiduNetdiskDownload\train1\dataset.yaml"  # 确保yaml中路径正确

# 验证数据集配置
def check_dataset():
    assert os.path.exists(dataset_yaml), f"dataset.yaml不存在: {dataset_yaml}"
    assert os.path.exists(r"E:\BaiduNetdiskDownload\train1\train"), "训练集路径错误"
    assert os.path.exists(r"E:\BaiduNetdiskDownload\train1\val"), "验证集路径错误"
    
    with open(dataset_yaml) as f:
        data = f.read()
        assert "names: ['holothurian', 'echinus', 'scallop', 'starfish']" in data, "类别名称不匹配"
    print("数据集验证通过！")

# 初始化模型
def setup_model():
    model = YOLO("yolov8n.yaml")  # 从零开始训练
    model.model.nc = 4  # 强制设置为4类水下生物
    return model

# 训练配置
def train_model():
    # 训练参数（针对水下生物优化）
    train_args = {
        'data': dataset_yaml,
        'epochs': 100,
        'batch': 16,
        'imgsz': 640,
        'device': '0',
        'optimizer': 'AdamW',
        'lr0': 0.01,
        'cos_lr': True,
        'patience': 20,
        'save_period': 10,
        'project': 'underwater_detection',
        'name': 'exp1',
        'exist_ok': True,
        'seed': 42,
        'pretrained': False,
        'workers': 4,
        'box': 7.5,  # 调整损失权重
        'cls': 1.5,  # 加强分类损失
        'dfl': 1.5,
        'verbose': True  # 显示详细训练日志
    }
    
    # 初始化模型
    model = setup_model()
    
    # 开始训练（带实时loss打印）
    print("\n开始训练...")
    results = model.train(**train_args)
    
    return model, results

# 实时loss监控
class LossTracker:
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
    
    def on_train_epoch_end(self, trainer):
        metrics = trainer.validator.metrics
        epoch = trainer.epoch
        train_loss = trainer.tloss
        val_loss = metrics.box_loss + metrics.cls_loss + metrics.dfl_loss
        
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        
        print(f"\nEpoch {epoch}/{trainer.epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Box Loss: {metrics.box_loss:.4f} | Cls Loss: {metrics.cls_loss:.4f} | DFL Loss: {metrics.dfl_loss:.4f}")

if __name__ == "__main__":
    # 验证数据集
    check_dataset()
    
    # 初始化loss跟踪器
    loss_tracker = LossTracker()
    
    # 训练模型
    model, results = train_model()
    
    # 保存loss曲线
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(loss_tracker.train_loss, label='Train Loss')
    plt.plot(loss_tracker.val_loss, label='Validation Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(r"E:\BaiduNetdiskDownload\train1\loss_curve.png")
    print(f"\nLoss曲线已保存到: E:\\BaiduNetdiskDownload\\train1\\loss_curve.png")
    
    # 打印最佳模型路径
    print(f"\n训练完成！最佳模型保存在:")
    print(f"{results.save_dir}\\weights\\best.pt")