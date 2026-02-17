import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit


class RobustOscillatorAnalyzer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.raw_time = []
        self.raw_position = []
        self.smooth_position = []
        self.fps = 0
        self.mm_per_pixel = 1.0

        # 结果参数
        self.beta = 0
        self.A0 = 0
        self.offset = 0
        self.energy_loss_ratio = 0

    def process_video_robust(self, roi_x_range=None, manual_scale=None):
        """
        核心处理逻辑：读取视频 -> CLAHE增强 -> 灰度重心法定位
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频: {self.video_path}")
            return

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 设置比例尺
        if manual_scale:
            self.mm_per_pixel = manual_scale
            print(f"使用手动比例尺: {self.mm_per_pixel} mm/pixel")
        else:
            print("⚠️ 警告：未设置比例尺，结果将以像素为单位。")

        print(f"开始分析 {total_frames} 帧数据...")

        # 1. 创建 CLAHE (自适应直方图均衡化) 对象，用于处理光照不均
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        while True:
            ret, frame = cap.read()
            if not ret: break

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # 2. ROI 裁剪 (只保留钢丝运动区域)
            if roi_x_range:
                process_frame = frame[:, roi_x_range[0]:roi_x_range[1]]
            else:
                process_frame = frame

            # 3. 转灰度并增强对比度
            gray = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY)
            enhanced_gray = clahe.apply(gray)

            # 4. 灰度重心法 (Grayscale Weighted Centroid)
            #    反转图像：钢丝变亮(高数值)，背景变黑(低数值)
            inverted = cv2.bitwise_not(enhanced_gray)

            #    截断底噪：把背景的浅灰色彻底变成黑色(0)，只保留明显的钢丝区域
            _, thresh_gray = cv2.threshold(inverted, 60, 255, cv2.THRESH_TOZERO)

            #    计算矩 (Moments)
            M = cv2.moments(thresh_gray)

            if M["m00"] != 0:
                # 重心 Y 坐标 = m01 / m00 (即使模糊也能算出亚像素中心)
                cY = M["m01"] / M["m00"]

                self.raw_time.append(timestamp)
                self.raw_position.append(cY * self.mm_per_pixel)
            else:
                # 这一帧没找到钢丝，跳过
                pass

        cap.release()

        # 数据转 numpy 数组
        self.raw_time = np.array(self.raw_time)
        self.raw_position = np.array(self.raw_position)

        # 数据去直流 (让平衡位置归零) 并取反 (物理坐标系向上为正，图像向下为正)
        if len(self.raw_position) > 0:
            # 这里取反是为了让波形符合直觉（向上运动为正），并减去均值
            self.raw_position = -(self.raw_position - np.mean(self.raw_position))
            # 进入平滑处理
            self._clean_and_smooth_data()
        else:
            print("❌ 错误：未能提取到任何有效数据。请检查 ROI 设置。")

    def _clean_and_smooth_data(self):
        """
        数据清洗：使用 Savitzky-Golay 滤波器去除抖动和噪点
        """
        if len(self.raw_position) < 10: return

        # 窗口长度设为帧率的一半左右（约0.5秒的数据窗），必须是奇数
        win_len = int(self.fps / 2)
        if win_len % 2 == 0: win_len += 1
        if win_len > len(self.raw_position): win_len = len(self.raw_position) // 2 * 2 + 1
        if win_len < 5: win_len = 5  # 最小窗口

        try:
            # polyorder=3 能很好地拟合正弦波的峰值
            self.smooth_position = savgol_filter(self.raw_position, window_length=win_len, polyorder=3)
            print("数据平滑处理完成。")
        except Exception as e:
            print(f"平滑处理失败 ({e})，使用原始数据。")
            self.smooth_position = self.raw_position

    def calculate_results(self):
        """计算物理参数：阻尼系数和能量损失"""
        if len(self.smooth_position) == 0: return

        # 1. 寻找波峰 (Peaks)
        # distance: 防止相邻帧重复检测
        peaks_idx, _ = find_peaks(self.smooth_position, distance=int(self.fps / 4), height=0)

        if len(peaks_idx) < 3:
            print("❌ 数据峰值不足，无法进行拟合。")
            return

        peak_t = self.raw_time[peaks_idx]
        peak_y = self.smooth_position[peaks_idx]

        # 2. 拟合阻尼包络线: A(t) = A0 * exp(-beta * t) + offset
        def envelope_func(t, A0, beta, offset):
            return A0 * np.exp(-beta * t) + offset

        try:
            # 初始猜测值
            p0 = [np.max(peak_y), 0.1, 0]
            popt, _ = curve_fit(envelope_func, peak_t, peak_y, p0=p0, maxfev=5000)
            self.A0, self.beta, self.offset = popt

            print("\n=== ✅ 分析结果 ===")
            print(f"阻尼系数 (β): {self.beta:.4f} s^-1")
        except:
            print("❌ 拟合失败。")
            return

        # 3. 计算能量损失 (E ∝ A^2)
        ratios = []
        for i in range(len(peak_y) - 1):
            loss = 1 - (peak_y[i + 1] / peak_y[i]) ** 2
            ratios.append(loss)
        if ratios:
            self.energy_loss_ratio = np.mean(ratios)
            print(f"平均周期能量损失率: {self.energy_loss_ratio:.2%}")

    def visualize(self):
        """可视化：对比原始与平滑数据，并展示拟合结果"""
        if len(self.smooth_position) == 0: return

        plt.figure(figsize=(12, 7))

        # 1. 原始数据（灰色细线）- 展示噪声情况
        plt.plot(self.raw_time, self.raw_position, color='lightgray', label='Raw Data (Noisy)', alpha=0.8)

        # 2. 平滑数据（蓝色实线）- 展示修复后效果
        plt.plot(self.raw_time, self.smooth_position, color='tab:blue', label='Smoothed Data (Savitzky-Golay)',
                 linewidth=1.5)

        # 3. 标记波峰
        peaks_idx, _ = find_peaks(self.smooth_position, distance=int(self.fps / 4), height=0)
        plt.plot(self.raw_time[peaks_idx], self.smooth_position[peaks_idx], "rx", markersize=8, label="Detected Peaks")

        # 4. 绘制拟合包络线
        if self.beta != 0:
            t_fit = np.linspace(self.raw_time[0], self.raw_time[-1], 200)
            y_fit = self.A0 * np.exp(-self.beta * t_fit) + self.offset
            plt.plot(t_fit, y_fit, "r--", linewidth=2, label=f"Envelope Fit (β={self.beta:.4f})")

            # 添加文本框
            info_text = (f"Damping Coeff ($\\beta$): {self.beta:.4f} $s^{{-1}}$\n"
                         f"Energy Loss/Cycle: {self.energy_loss_ratio:.2%}")
            plt.gca().text(0.02, 0.95, info_text, transform=plt.gca().transAxes,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.title(f"Damped Oscillation Analysis (Method: Weighted Centroid + CLAHE)")
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement (mm)")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# --- 主程序入口 ---
if __name__ == "__main__":
    # 1. 指定视频文件 (使用刚才生成的模拟视频)
    video_file = "simulated_experiment.mp4"

    analyzer = RobustOscillatorAnalyzer(video_file)

    # 2. 运行处理
    # roi_x_range=(150, 350): 仅关注水平方向 150-350 像素范围（排除左右背景）
    # manual_scale=0.2: 之前生成视频时 5 pixel = 1 mm, 所以 1/5 = 0.2 mm/pixel
    analyzer.process_video_robust(roi_x_range=(150, 350), manual_scale=0.2)

    # 3. 计算结果
    analyzer.calculate_results()

    # 4. 画图
    analyzer.visualize()