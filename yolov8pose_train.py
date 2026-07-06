from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"]= 'TRUE'
# 加载预训练的姿态估计模型
# 可选模型规格: yolov8n-pose.pt (nano), yolov8s-pose.pt (small),
#          yolov8m-pose.pt (medium), yolov8l-pose.pt (large), yolov8x-pose.pt (x-large)
# model = YOLO('yolov8n-pose.pt')  # 从预训练权重开始训练

# 也可以从头开始训练（不推荐，除非有大量数据）
# model = YOLO('yolov8-pose.yaml')
model = YOLO('yolo26-pose.yaml')

# 开始训练
results = model.train(
    data=r'D:\xinyu\ultralytics-main\datasets\tip-traindata0608\data.yaml' ,  # 数据集配置文件路径
    epochs=100,  # 训练轮数
    imgsz=480,  # 输入图像尺寸
    batch=16,  # 批大小（可根据GPU内存调整）
    device=0,  # GPU设备，如 'cpu' 或 '0,1' 多GPU
    workers=0,  # 数据加载线程数

    # 其他设置
    seed=42,  # 随机种子
    resume=False,  # 是否从断点恢复训练
    project='tip_pose扩增_26pose',  # 结果保存目录
    name='100e_0609_01',  # 实验名称
    exist_ok=False,  # 是否覆盖已有结果

    # 损失权重值
    # 默认
    box=7.5,  # 降低边界框损失权重
    pose=12.0,  # 提高关键点损失权重
    kobj=1.0, # 也可以适当提高关键点置信度损失

    # box=3.0,  # 降低边界框损失权重
    # pose=20.0,  # 提高关键点损失权重
    # kobj=2.0,  # 也可以适当提高关键点置信度损失

# ========== 数据增强参数 ==========
#     # 颜色空间增强
#     hsv_h=0.015,              # 色调扰动范围（HSV-H），默认0.015
#     hsv_s=0.7,                # 饱和度扰动范围，默认0.7
#     hsv_v=0.4,                # 明度扰动范围，默认0.4

    # # 几何增强
    # degrees=0.0,              # 旋转角度（度），0表示不旋转，可设为±10等
    translate=0.0,            # 平移比例（相对于图像尺寸），默认0.1
    scale=0.0,                # 缩放比例（例如0.5表示随机缩放0.5~1.5倍），默认0.5
    # shear=0.0,                # 剪切变换（度），默认0.0
    # perspective=0.0,          # 透视变换（系数），默认0.0
    #
    # # 翻转增强
    # flipud=0.0,               # 垂直翻转概率，默认0.0
    fliplr=0.0,               # 水平翻转概率，默认0.5
    #
    # # 混合增强
    mosaic=0.0,               # 马赛克增强概率（4张图拼成1张），默认1.0
    # mixup=0.0,                # 混合增强概率（两张图线性混合），默认0.0
    # copy_paste=0.0,           # 复制粘贴增强概率（实例分割用），姿态估计中一般设为0
    #
    # # 其他增强
    # erasing=0.4,              # 随机擦除概率（Cutout），默认0.4
    # # crop_fraction=1.0,        # 裁剪比例（用于分类任务，姿态估计通常保持默认）

)