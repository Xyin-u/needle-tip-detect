import csv
from copy import deepcopy
from pathlib import Path

import cv2

from ultralytics import YOLO

# ========== 配置参数 ==========
model = YOLO(r"D:\xinyu\ultralytics-main\runs\pose\tip_pose\0526\weights\best.pt")
input_dir = Path(r"C:\Users\tiger\Desktop\tip_dataset")
output_dir = Path(r"D:\xinyu\ultralytics-main\runs\pose\imgsandlabels")
conf_threshold = 0.5  # 置信度阈值，低于此值认为定位失败
kpt_radius = 2  # 红点半径
output_dir.mkdir(exist_ok=True)

# 存储 CSV 行数据
csv_rows = []

# 遍历所有图片
for img_path in input_dir.glob("*.*"):
    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    # 预测
    results = model(img_path, conf=conf_threshold)
    result = results[0]

    # 复制原图用于绘制
    img = deepcopy(result.orig_img)
    h, w = img.shape[:2]  # 获取图像尺寸（用于归一化）

    # 默认值（定位失败）
    success = False
    tip_x, tip_y = -1, -1
    confidence = 0.0

    # ===== 保存 YOLO Pose 标签文件（.txt） =====
    txt_path = input_dir / (img_path.stem + ".txt")  # 保存在图片同一目录
    if result.boxes is not None and result.keypoints is not None and len(result.boxes) > 0:
        # 取第一个检测结果（可根据需要遍历所有）
        box = result.boxes.data[0].cpu().numpy()  # [x1, y1, x2, y2, conf, cls]
        kpts = result.keypoints.data[0].cpu().numpy()  # (num_kpts, 3)  [x, y, visible_conf]

        # 提取边界框坐标（绝对值）
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        # 转换为中心点宽高并归一化
        cx = (x1 + x2) / 2.0 / w
        cy = (y1 + y2) / 2.0 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        class_id = int(box[5])

        # 构建关键点字符串：每个关键点 x_norm y_norm visible
        kpt_str_list = []
        for kp in kpts:
            kp_x_norm = kp[0] / w
            kp_y_norm = kp[1] / h
            kp_conf = kp[2]
            # visible: 2 表示存在且可见（置信度≥阈值），0 表示不可见或未检测到
            visible = 2 if kp_conf >= conf_threshold else 0
            kpt_str_list.append(f"{kp_x_norm:.6f} {kp_y_norm:.6f} {visible}")

        # 组合成一行标签
        label_line = f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} " + " ".join(kpt_str_list)
        with open(txt_path, "w") as f:
            f.write(label_line)
        print(f"标签已保存: {txt_path}")
    else:
        # 如果没有检测到任何目标，生成空文件（或跳过）
        # 这里选择生成空文件，表示该图无标注
        with open(txt_path, "w") as f:
            pass
        print(f"未检测到目标，生成空标签: {txt_path}")

    # 处理关键点（用于绘制和CSV）
    if result.keypoints is not None and len(result.keypoints.data) > 0:
        # 取第一个检测目标的第一个关键点（模型只有一个关键点）
        kpts = result.keypoints.data[0]  # shape: (num_kpts, 3)
        if len(kpts.shape) == 2:
            kpt = kpts[0]  # (x, y, conf)
        else:
            kpt = kpts  # 已经是 (3,)
        x, y, conf = kpt.cpu().numpy() if hasattr(kpt, "cpu") else kpt
        tip_x, tip_y = int(x), int(y)
        confidence = float(conf)

        # 判断是否成功（置信度足够且坐标在图像范围内可选）
        if confidence >= conf_threshold:
            success = True
            # 画红色圆点
            cv2.circle(img, (tip_x, tip_y), kpt_radius, (0, 0, 255), -1)
            # 显示坐标文本
            cv2.putText(
                img,
                f"({tip_x},{tip_y})",
                (tip_x + 5, tip_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        else:
            print(f"  置信度不足 ({conf:.2f} < {conf_threshold})，不绘制")
    else:
        print("  未检测到关键点")

    # 记录 CSV 行
    csv_rows.append(
        {"文件名": img_path.name, "x坐标": tip_x, "y坐标": tip_y, "定位成功": success, "置信度": confidence}
    )

    # 保存带标注的图片
    out_path = output_dir / img_path.name
    cv2.imwrite(str(out_path), img)
    print(f"完成: {img_path.name}  坐标: ({tip_x},{tip_y})  成功: {success}")

# 写入 CSV 文件
if csv_rows:
    csv_path = output_dir / "pin_tip_coords.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["文件名", "x坐标", "y坐标", "定位成功", "置信度"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n坐标已保存至: {csv_path}")
else:
    print("未生成任何结果")
