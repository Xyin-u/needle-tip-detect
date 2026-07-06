from ultralytics import YOLO
from copy import deepcopy
import cv2
from pathlib import Path
import csv
import numpy as np

# ========== 配置参数 ==========
model = YOLO(r'D:\xinyu\ultralytics-main\runs\pose\tip_pose扩增_v8pose\25e_0601_0_wl\weights\best.pt')
input_dir = Path(r'D:\xinyu\ultralytics-main\datasets\tip-traindata0601\images\test')
labels_dir = Path(r'D:\xinyu\ultralytics-main\datasets\tip-traindata0601\labels\test')  # 存放真实标注的目录（YOLO格式txt）
output_dir = Path(r'D:\xinyu\ultralytics-main\runs\pose\tip-kuozeng_data0601_0_wloss_test')
conf_threshold = 0.5          # 置信度阈值
kpt_radius = 1                # 关键点红点半径
box_thickness = 1             # 检测框线条粗细
output_dir.mkdir(exist_ok=True)

# 存储 CSV 行数据
csv_rows = []

# 用于计算平均像素误差的列表
pixel_errors = []   # 存储每个有效样本的欧氏距离（像素）

# 遍历所有图片
for img_path in input_dir.glob('*.*'):
    if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
        continue

    # 预测
    results = model(img_path, conf=conf_threshold, imgsz=480)
    result = results[0]

    # 复制原图用于绘制
    img = deepcopy(result.orig_img)
    h, w = img.shape[:2]

    # 默认值（定位失败）
    success = False
    tip_x, tip_y = -1, -1
    confidence = 0.0

    # 1. 绘制检测框（如果存在）
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            box_conf = float(box.conf[0]) if box.conf is not None else 0.0
            cls_id = int(box.cls[0]) if box.cls is not None else -1
            class_name = model.names.get(cls_id, "needle") if hasattr(model, 'names') else "needle"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), box_thickness)
            label = f"{class_name} {box_conf:.2f}"
            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    else:
        print(f"  未检测到检测框")

    # 2. 处理关键点
    if result.keypoints is not None and len(result.keypoints.data) > 0:
        kpts = result.keypoints.data[0]          # 取第一个目标的关键点
        if len(kpts.shape) == 2:
            kpt = kpts[0]                        # (x, y, conf)
        else:
            kpt = kpts
        x, y, conf = kpt
        tip_x, tip_y = int(x), int(y)
        confidence = float(conf)

        if confidence >= conf_threshold:
            success = True
            cv2.circle(img, (tip_x, tip_y), kpt_radius, (0, 0, 255), -1)
            cv2.putText(img, f"({tip_x},{tip_y})", (tip_x+5, tip_y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            print(f"  关键点置信度不足 ({conf:.2f} < {conf_threshold})，不绘制关键点")
    else:
        print(f"  未检测到关键点")

    # ========== 计算像素误差（如果成功且有真实标注） ==========
    # 查找对应的真实标注文件
    label_path = labels_dir / (img_path.stem + '.txt')
    gt_x, gt_y = -1, -1
    gt_visible = False
    if label_path.exists():
        with open(label_path, 'r') as f:
            line = f.readline().strip()
            if line:
                parts = line.split()
                # 格式: class xc_norm yc_norm w_norm h_norm kpt_x_norm kpt_y_norm visibility ...
                # 假设只有一个关键点，且位置在第6、7个字段（索引5,6）
                if len(parts) >= 8:
                    kpt_x_norm = float(parts[5])
                    kpt_y_norm = float(parts[6])
                    visibility = int(parts[7])   # 2=可见且标记, 1=遮挡, 0=未标记
                    if visibility >= 1:          # 认为可见或遮挡都算有效（可根据需要调整）
                        gt_x = int(kpt_x_norm * w)
                        gt_y = int(kpt_y_norm * h)
                        gt_visible = True
    else:
        print(f"  警告: 未找到标注文件 {label_path}，跳过误差计算")

    # 如果模型预测成功且真实关键点有效，计算欧氏距离
    if success and gt_visible:
        dist = np.sqrt((tip_x - gt_x)**2 + (tip_y - gt_y)**2)
        pixel_errors.append(dist)
        # 可选：在图像上绘制真实点（蓝色）用于对比
        cv2.circle(img, (gt_x, gt_y), kpt_radius, (255, 0, 0), -1)
        cv2.putText(img, f"GT({gt_x},{gt_y})", (gt_x+5, gt_y+25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        print(f"  像素误差: {dist:.2f} px (预测: {tip_x},{tip_y} 真实: {gt_x},{gt_y})")
    elif success and not gt_visible:
        print(f"  真实关键点无效(不可见)，无法计算误差")
    elif not success and gt_visible:
        print(f"  模型未检测到关键点，但真实存在 -> 漏检")
        # 也可以记录一个很大的误差（例如 None），这里选择不计入平均

    # 记录 CSV 行（增加真实坐标和误差列）
    csv_rows.append({
        "文件名": img_path.name,
        "x坐标": tip_x,
        "y坐标": tip_y,
        "定位成功": success,
        "置信度": confidence,
        "真实x": gt_x if gt_visible else -1,
        "真实y": gt_y if gt_visible else -1,
        "像素误差": f"{dist:.2f}" if success and gt_visible else "N/A"
    })

    # 保存带标注的图片
    out_path = output_dir / img_path.name
    cv2.imwrite(str(out_path), img)
    print(f"完成: {img_path.name}  坐标: ({tip_x},{tip_y})  成功: {success}")

# 写入 CSV 文件
if csv_rows:
    csv_path = output_dir / "pin_tip_coords.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        fieldnames = ["文件名", "x坐标", "y坐标", "定位成功", "置信度", "真实x", "真实y", "像素误差"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n坐标已保存至: {csv_path}")

# 输出平均像素误差
if pixel_errors:
    mean_error = np.mean(pixel_errors)
    std_error = np.std(pixel_errors)
    print(f"\n========== 平均像素误差评估 ==========")
    print(f"有效样本数: {len(pixel_errors)}")
    print(f"平均像素误差: {mean_error:.2f} px")
    print(f"标准差: {std_error:.2f} px")
    print(f"最大误差: {max(pixel_errors):.2f} px")
    print(f"最小误差: {min(pixel_errors):.2f} px")
    # 同时将统计信息保存到文本文件
    stats_path = output_dir / "pixel_error_stats.txt"
    with open(stats_path, 'w') as f:
        f.write(f"有效样本数: {len(pixel_errors)}\n")
        f.write(f"平均像素误差: {mean_error:.2f} px\n")
        f.write(f"标准差: {std_error:.2f} px\n")
        f.write(f"最大误差: {max(pixel_errors):.2f} px\n")
        f.write(f"最小误差: {min(pixel_errors):.2f} px\n")
    print(f"统计信息已保存至: {stats_path}")
else:
    print("\n没有足够的数据计算平均像素误差（可能没有同时具备预测成功和真实标注的样本）")