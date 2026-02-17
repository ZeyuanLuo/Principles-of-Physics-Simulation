import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


class DampedOscillationAnalyzer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.time_data = []
        self.position_data_mm = []
        self.fps = 0
        self.mm_per_pixel = 1.0  # 默认值，稍后计算
        # 阻尼拟合结果
        self.beta = 0
        self.A0 = 0
        self.offset = 0
        self.energy_loss_ratio = 0

    # ===========================
    # 第一部分：视频处理与追踪方案
    # ===========================

    def _calibrate_scale_automatic(self, frame):
        """
        尝试自动识别背景刻度线来计算 mm/pixel 比例尺。
        注意：此方法对光照和刻度清晰度很敏感。如果失败，建议手动指定比例。
        """
        print("正在尝试自动标定刻度...")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # 霍夫直线变换检测水平线
        # minLineLength: 线段最小长度，maxLineGap: 线段断裂最大间隔
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=100, maxLineGap=10)

        if lines is None or len(lines) < 5:
            print("警告：未能检测到足够的刻度线，自动标定失败。将使用默认比例 1.0。")
            return 1.0

        # 提取水平线的 Y 坐标
        y_coords = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 5:  # 确保是大致水平的线
                y_coords.append(y1)

        y_coords.sort()

        # 计算相邻线的间距
        diffs = np.diff(y_coords)
        # 过滤掉太近的重影线和太远的间隔
        valid_diffs = diffs[(diffs > 5) & (diffs < 100)]

        if len(valid_diffs) > 0:
            # 取中位数作为最可能的 1mm 刻度间距像素值
            avg_pixel_dist = np.median(valid_diffs)
            # 假设背景最小刻度为 1mm
            mm_per_pixel = 1.0 / avg_pixel_dist
            print(f"自动标定成功: 1mm 约等于 {avg_pixel_dist:.2f} 像素。比例尺: {mm_per_pixel:.4f} mm/pixel")
            return mm_per_pixel
        else:
            print("警告：无法确定可靠的刻度间距。将使用默认比例。")
            return 1.0

    def process_video(self, roi_x_range=None, manual_mm_per_pixel=None):
        """
        主处理循环：读取视频，追踪钢丝位置。
        :param roi_x_range: 元组 (x_start, x_end)，用于裁剪感兴趣区域，减少背景干扰。
        :param manual_mm_per_pixel: 如果自动标定不准，可手动传入比例尺值。
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {self.video_path}")

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        ret, first_frame = cap.read()
        if not ret: return

        # 1. 标定比例尺
        if manual_mm_per_pixel:
            self.mm_per_pixel = manual_mm_per_pixel
            print(f"使用手动设置的比例尺: {self.mm_per_pixel} mm/pixel")
        else:
            self.mm_per_pixel = self._calibrate_scale_automatic(first_frame)

        print("开始视频追踪处理...")
        self.time_data = []
        self.position_data_mm = []

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 回到开头

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # 秒

            # --- 图像识别核心 ---
            # 1. ROI 裁剪 (如果指定)
            process_frame = frame
            offset_x = 0
            if roi_x_range:
                offset_x = roi_x_range[0]
                process_frame = frame[:, roi_x_range[0]:roi_x_range[1]]

            # 2. 转灰度
            gray = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY)

            # 3. 二值化
            # 背景白，钢丝黑。使用 THRESH_BINARY_INV 反转，使钢丝变成白色区域方便寻找
            threshold_value = 60  # 这个阈值可能需要根据光照调整
            _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

            # 可选：形态学操作去噪点 (腐蚀再膨胀)
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            # 4. 寻找轮廓
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            needle_y_pixel = None
            if contours:
                # 假设最大的轮廓是我们的钢丝指针
                largest_contour = max(contours, key=cv2.contourArea)

                # 忽略太小的噪点
                if cv2.contourArea(largest_contour) > 50:
                    # 5. 计算质心 (获得亚像素精度)
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        # cX = int(M["m10"] / M["m00"]) + offset_x
                        cY = M["m01"] / M["m00"]  # 使用浮点数以提高精度
                        needle_y_pixel = cY

            # 6. 存储数据
            if needle_y_pixel is not None:
                # 注意：图像坐标系 Y 轴向下为正，物理坐标系通常向上为正。
                # 这里我们暂且保留图像坐标方向，后续分析时再统一。
                self.time_data.append(current_time)
                # 将像素坐标转换为毫米坐标
                self.position_data_mm.append(needle_y_pixel * self.mm_per_pixel)

                # (可选) 在视频上实时显示追踪结果用于调试
                # display_frame = frame.copy()
                # cv2.circle(display_frame, (int(frame.shape[1]/2), int(needle_y_pixel)), 5, (0, 0, 255), -1)
                # cv2.imshow('Tracking Debug', display_frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        # cv2.destroyAllWindows()
        self.time_data = np.array(self.time_data)
        # 让数据中心化（去直流分量，使平衡位置在 y=0 附近）
        self.position_data_mm = np.array(self.position_data_mm)
        self.position_data_mm = -(self.position_data_mm - np.mean(self.position_data_mm))
        print(f"视频处理完成，共捕获 {len(self.time_data)} 帧数据。")

    # ===========================
    # 第二部分：自动计算阻尼与能量
    # ===========================

    def _damped_sine_envelope(self, t, A0, beta, offset):
        """阻尼振动的包络线方程: A(t) = A0 * exp(-beta * t) + offset"""
        return A0 * np.exp(-beta * t) + offset

    def analyze_data(self):
        """自动寻找峰值并计算阻尼系数"""
        if len(self.time_data) == 0:
            print("错误：没有数据可分析。请先运行 process_video。")
            return

        # 1. 寻找波峰 (Peaks)
        # distance参数防止检测到相邻的假峰，设置为采样率的1/5是一个经验值
        min_distance = int(self.fps / 5)
        # height参数确保只找正半轴的峰值
        peaks_idx, _ = find_peaks(self.position_data_mm, distance=min_distance, height=0)

        self.peak_times = self.time_data[peaks_idx]
        self.peak_values = self.position_data_mm[peaks_idx]

        if len(self.peak_times) < 3:
            print("错误：检测到的峰值太少，无法进行可靠拟合。")
            return

        # 2. 拟合指数衰减包络线
        # 初始猜测值 [A0, beta, offset]
        p0 = [np.max(self.peak_values), 0.1, 0]
        try:
            # 使用 scipy.optimize.curve_fit 进行非线性最小二乘拟合
            popt, pcov = curve_fit(self._damped_sine_envelope, self.peak_times, self.peak_values, p0=p0, maxfev=5000)
            self.A0, self.beta, self.offset = popt
            print("\n=== 分析结果 ===")
            print(f"拟合方程: A(t) = {self.A0:.3f} * exp(-{self.beta:.4f} * t) + {self.offset:.3f}")
            print(f"阻尼系数 (β): {self.beta:.5f} s^-1")
        except RuntimeError:
            print("错误：指数拟合失败。数据可能太过嘈杂。")
            return

        # 3. 计算能量损耗
        # 能量与振幅平方成正比 E ∝ A^2
        # 计算相邻峰值间的平均能量损失率
        if len(self.peak_values) >= 2:
            energy_ratios = []
            for i in range(len(self.peak_values) - 1):
                A_n = self.peak_values[i]
                A_n1 = self.peak_values[i + 1]
                # 能量比 E_{n+1} / E_n = (A_{n+1} / A_n)^2
                ratio = (A_n1 / A_n) ** 2
                energy_ratios.append(ratio)

            avg_energy_ratio = np.mean(energy_ratios)
            self.energy_loss_ratio = 1.0 - avg_energy_ratio
            print(f"平均周期能量保留率 (E_n+1 / E_n): {avg_energy_ratio:.2%}")
            print(f"平均周期能量损失率: {self.energy_loss_ratio:.2%}")

    # ===========================
    # 第三部分：可视化
    # ===========================

    def visualize(self):
        """绘制振动曲线、检测到的峰值和拟合的包络线"""
        if len(self.time_data) == 0 or self.beta == 0:
            print("数据不完整，无法绘图。")
            return

        plt.figure(figsize=(12, 6))

        # 1. 绘制原始振动数据
        plt.plot(self.time_data, self.position_data_mm, 'b-', alpha=0.5, label='Raw Vibration Data (mm)')

        # 2. 绘制识别出的波峰
        plt.plot(self.peak_times, self.peak_values, 'rx', markersize=10, label='Detected Peaks')

        # 3. 绘制拟合的指数衰减包络线
        t_fit = np.linspace(self.time_data[0], self.time_data[-1], 500)
        A_fit = self._damped_sine_envelope(t_fit, self.A0, self.beta, self.offset)
        plt.plot(t_fit, A_fit, 'r--', linewidth=2, label=f'Envelope Fit (β={self.beta:.4f})')

        plt.title('Damped Harmonic Oscillation Analysis')
        plt.xlabel('Time (s)')
        plt.ylabel('Displacement (mm)')
        plt.grid(True)
        plt.legend()

        # 在图上添加文本信息
        info_text = f"Damping Coeff (β): {self.beta:.5f} $s^{{-1}}$\nEnergy Loss/Cycle: {self.energy_loss_ratio:.2%}"
        plt.text(0.02, 0.95, info_text, transform=plt.gca().transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        print("显示图表...")
        plt.show()


# ===========================
# 主程序运行示例
# ===========================
if __name__ == "__main__":
    # 【重要】请将此处替换为您的实际视频文件路径
    video_file = "simulated_experiment.mp4"

    # 如果您没有视频，请先录制一个。
    # 要求：背景有清晰毫米刻度，黑色横向钢丝在前景上下振动。

    try:
        analyzer = DampedOscillationAnalyzer(video_file)

        # --- 步骤 1: 处理视频 ---
        # roi_x_range: 用于横向裁剪视频。例如，只关注图像中间 x=300 到 x=600 的区域。
        # 如果自动标定失败，可以手动提供 manual_mm_per_pixel (例如: 0.1 mm/pixel)
        # analyzer.process_video(roi_x_range=(300, 600))
        # 下面是使用自动标定和全画面的示例：
        analyzer.process_video()

        # --- 步骤 2: 数据分析计算 ---
        analyzer.analyze_data()

        # --- 步骤 3: 可视化结果 ---
        analyzer.visualize()

    except FileNotFoundError:
        print(f"找不到视频文件: {video_file}，请检查路径。")
    except Exception as e:
        print(f"发生错误: {e}")