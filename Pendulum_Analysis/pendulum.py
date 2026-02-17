"""
文件名: pendulum.py
功能:
1. OpenCV 视觉追踪 (从视频提取坐标)
2. 信号处理 (低通滤波去噪)
3. 动力学反演与验证 (用户输入参数 vs 视频数据)
4. 绘制相图与庞加莱截面
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.integrate import odeint
from tqdm import tqdm

# ===========================
# 核心工具类
# ===========================

class MotionTracker:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"无法打开视频: {video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def track_ball(self):
        """简单的由暗到亮的阈值追踪法"""
        print(f"--- 开始视觉追踪: {self.frame_count} 帧 ---")
        positions = []

        with tqdm(total=self.frame_count) as pbar:
            while True:
                ret, frame = self.cap.read()
                if not ret: break

                # 1. 灰度与二值化
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

                # 2. 寻找最大轮廓
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        positions.append([cx, cy])
                    else:
                        positions.append([np.nan, np.nan])
                else:
                    positions.append([np.nan, np.nan])
                pbar.update(1)

        self.cap.release()
        return np.array(positions)

class SignalProcessor:
    @staticmethod
    def butter_lowpass_filter(data, cutoff, fs, order=4):
        """零相位低通滤波器"""
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        # 填充NaN值
        mask = np.isnan(data)
        data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])
        return filtfilt(b, a, data)

    @staticmethod
    def calculate_angle(xy_data, pivot_estimate=None):
        """将像素坐标转换为物理角度"""
        # 如果未提供支点，假设支点在摆动轨迹圆拟合圆心，或者简单取最高点上方
        # 这里简化处理：假设视频生成时支点在画面上方居中 (用户需校准，或自动检测)
        if pivot_estimate is None:
            # 自动估算：取X的平均值作为Pivot X，取最小Y(最高点)减去一部分作为Pivot Y
            pivot_x = np.nanmean(xy_data[:, 0])
            pivot_y = np.nanmin(xy_data[:, 1]) - 100 # 粗略估计
            pivot_estimate = (pivot_x, pivot_y)

        dx = xy_data[:, 0] - pivot_estimate[0]
        dy = xy_data[:, 1] - pivot_estimate[1]
        # 注意: atan2(x, y) 对应 0度在垂直向下(Y轴正向)
        theta = np.arctan2(dx, dy)
        return theta

class AnalysisPlotter:
    def __init__(self, t, theta_raw, theta_filt, omega_filt, sim_theta, sim_omega):
        self.t = t
        self.theta_raw = theta_raw
        self.theta_filt = theta_filt
        self.omega_filt = omega_filt
        self.sim_theta = sim_theta
        self.sim_omega = sim_omega

    def show_dashboard(self, T_natural):
        fig = plt.figure(figsize=(14, 10))

        # 1. 滤波效果对比
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.set_title("Filter Performance (Data Cleaning)")
        ax1.plot(self.t[:300], np.degrees(self.theta_raw[:300]), 'r.', alpha=0.2, label='Raw Vision Data')
        ax1.plot(self.t[:300], np.degrees(self.theta_filt[:300]), 'b-', linewidth=1.5, label='Low-pass Filtered')
        ax1.set_ylabel('Angle (deg)')
        ax1.legend()
        ax1.grid(True)

        # 2. 实验 vs 理论模型
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.set_title("Experiment (Video) vs Theory (Model)")
        ax2.plot(self.t, np.degrees(self.theta_filt), 'b-', label='Video Tracked', alpha=0.7)
        ax2.plot(self.t, np.degrees(self.sim_theta), 'k--', label='Theoretical Model', alpha=0.7)
        ax2.set_xlabel('Time (s)')
        ax2.legend()
        ax2.grid(True)

        # 3. 庞加莱截面 (Poincaré Section)
        ax3 = fig.add_subplot(2, 2, 4)
        ax3.set_title(f"Poincaré Map (Sampled @ T={T_natural:.2f}s)")

        # 绘制背景相图轨迹
        ax3.plot(np.degrees(self.theta_filt), self.omega_filt, 'g-', alpha=0.1)

        # 庞加莱采样
        indices = []
        max_idx = len(self.t)
        dt = self.t[1] - self.t[0]
        step = int(T_natural / dt)

        for i in range(0, max_idx, step):
            indices.append(i)

        p_theta = self.theta_filt[indices]
        p_omega = self.omega_filt[indices]

        # 颜色映射表示时间流逝 (浅红 -> 深红)
        ax3.scatter(np.degrees(p_theta), p_omega, c=np.arange(len(indices)), cmap='Reds', edgecolors='k', s=40, zorder=5)
        ax3.set_xlabel('Theta (deg)')
        ax3.set_ylabel('Omega (rad/s)')
        ax3.grid(True)

        # 4. 完整相图
        ax4 = fig.add_subplot(2, 2, 3)
        ax4.set_title("Phase Portrait (Filtered Data)")
        ax4.plot(np.degrees(self.theta_filt), self.omega_filt, 'purple', linewidth=0.5)
        ax4.set_xlabel('Theta (deg)')
        ax4.set_ylabel('Omega (rad/s)')
        ax4.grid(True)

        plt.tight_layout()
        plt.show()

# ===========================
# 主程序
# ===========================

def main():
    video_file = "pendulum_experiment.mp4"

    # 1. 执行视觉追踪
    tracker = MotionTracker(video_file)
    raw_pos = tracker.track_ball()

    # 2. 手动输入参数 (模拟实验后的数据处理环节)
    print("\n--- 请输入参数进行动力学建模验证 ---")
    # 为了演示方便，这里预设了和生成时一样的参数，实际使用可改为 input()
    L_input = 1.0
    m_input = 1.0
    c_input = 0.05 # 尝试输入 0.04 看看和视频的差异

    # 3. 数据处理 (坐标 -> 角度 -> 滤波)
    # 假设已知视频中的支点位置 (通常由生成器告知或手动标定)
    # 这里的 (400, 150) 是 generate_data.py 里的默认支点 (width//2, height//4)
    pivot_pos = (400, 150)

    processor = SignalProcessor()
    theta_raw = processor.calculate_angle(raw_pos, pivot_pos)

    # 低通滤波: 截止频率 5Hz (去除图像抖动噪声)
    theta_filt = processor.butter_lowpass_filter(theta_raw, cutoff=5.0, fs=tracker.fps)

    # 计算角速度 (差分)
    dt = 1.0 / tracker.fps
    omega_filt = np.gradient(theta_filt, dt)
    # 对角速度再次轻微滤波，因为微分会放大噪声
    omega_filt = processor.butter_lowpass_filter(omega_filt, cutoff=3.0, fs=tracker.fps)

    # 4. 理论模型仿真 (用于对比)
    # 定义方程
    I = m_input * L_input**2
    d = L_input
    def model(y, t):
        th, w = y
        return [w, (-m_input * 9.81 * d * np.sin(th) - c_input * w) / I]

    t_span = np.linspace(0, len(theta_filt)*dt, len(theta_filt))
    # 使用滤波后的初始状态作为仿真起点
    y_sim = odeint(model, [theta_filt[0], omega_filt[0]], t_span)

    # 5. 绘制结果
    T_natural = 2 * np.pi * np.sqrt(L_input / 9.81)
    plotter = AnalysisPlotter(t_span, theta_raw, theta_filt, omega_filt, y_sim[:,0], y_sim[:,1])
    plotter.show_dashboard(T_natural)

if __name__ == "__main__":
    main()