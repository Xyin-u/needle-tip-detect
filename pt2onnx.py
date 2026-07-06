from ultralytics import YOLO

# Load a model
model = YOLO(
    r"D:\xinyu\ultralytics-main\runs\pose\tip_pose扩增_v8pose\200e_0608_wl_02\weights\best.pt"
)  # load a custom-trained model

# Export the model
model.export(format="onnx", imgsz=480, half=False)
