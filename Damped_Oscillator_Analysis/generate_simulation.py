import cv2
import numpy as np
import math


def generate_damped_video(filename='simulated_experiment.mp4', duration=10, fps=60):
    # --- 1. 视频与物理参数设置 ---
    width, height = 640, 800
    pixels_per_mm = 5.0  # 设定比例尺：5个像素代表1mm

    # 物理模型参数: y(t) = A * e^(-beta*t) * cos(omega*t)
    amplitude_mm = 50.0  # 初始振幅 50mm
    beta = 0.3  # 阻尼系数
    omega = 2 * math.pi * 1.0  # 角频率 (1Hz)
    center_y_mm = 80.0  # 平衡位置在刻度尺的 80mm 处

    # 视频编码器设置
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    print(f"开始生成模拟视频: {filename}")
    print(f"物理参数: 阻尼系数 beta={beta}, 频率=1Hz")

    # --- 2. 预先生成背景（绘制刻度尺）---
    # 创建白底
    background = np.full((height, width, 3), 255, dtype=np.uint8)

    # 绘制刻度线
    # 我们假设从图像顶部开始就是刻度 0mm
    max_mm = int(height / pixels_per_mm)

    for mm in range(0, max_mm):
        y_pos = int(mm * pixels_per_mm)

        # 区分长短刻度
        if mm % 10 == 0:  # 厘米刻度 (长线)
            cv2.line(background, (100, y_pos), (250, y_pos), (150, 150, 150), 2)
            # 添加数字标签
            cv2.putText(background, f"{mm}", (50, y_pos + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        else:  # 毫米刻度 (短线)
            cv2.line(background, (100, y_pos), (180, y_pos), (200, 200, 200), 1)

    # --- 3. 逐帧生成动画 ---
    total_frames = duration * fps

    for i in range(total_frames):
        t = i / fps

        # 复制背景，避免每帧重画刻度
        frame = background.copy()

        # 计算当前物理位移 (mm)
        displacement = amplitude_mm * math.exp(-beta * t) * math.cos(omega * t)

        # 计算当前指针的绝对位置 (mm -> pixel)
        # 注意：OpenCV中 y轴向下为正，所以用 center + displacement
        current_y_mm = center_y_mm + displacement
        current_y_pixel = int(current_y_mm * pixels_per_mm)

        # 绘制黑色钢丝 (指针)
        # 模拟钢丝有一定的粗细 (thickness=3)
        # 横跨 x=150 到 x=350 的区域
        cv2.line(frame, (150, current_y_pixel), (350, current_y_pixel), (0, 0, 0), 4)

        # (可选) 添加一些随机噪声，模拟真实摄像头的噪点
        noise = np.random.normal(0, 2, frame.shape).astype(np.uint8)
        # 这里为了演示清晰暂不加噪声，如果想更真实可以取消下面注释
        # frame = cv2.add(frame, noise)

        out.write(frame)

        if i % 60 == 0:
            print(f"进度: {t:.1f}s / {duration}s")

    out.release()
    print("视频生成完毕！文件保存在当前目录。")


if __name__ == "__main__":
    generate_damped_video()