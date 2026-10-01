import os

from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ===================== 1. 加载训练好的模型 =====================
# 这里加载你训练完的最佳权重
# 权重路径：tip_pose/300e_0527/weights/best.pt
model = YOLO("runs/pose/tip_pose/0526/weights/best.pt")

# ===================== 2. 在验证集上评估模型 =====================
# 自动跑一遍验证集，输出 AP、精度、召回率等指标
metrics = model.val(
    data=r"D:\xinyu\ultralytics-main\datasets\data.yaml",
    imgsz=640,
    batch=4,
    device=0,
    workers=0,
    name="val_results",  # 验证结果保存文件夹
)
