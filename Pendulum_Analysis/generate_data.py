"""
文件名: generate_data.py
功能:
1. 基于物理方程模拟单摆/复摆运动
2. 渲染生成高帧率合成视频 (.mp4)
3. 保存真实轨迹数据 (.npz) 用于后续验证
"""

import cv2
import numpy as np
from scipy.integrate import odeint
from tqdm import tqdm


class PendulumSimulator:
    def __init__(self, m=1.0, c=0.1, L=1.0, I_pivot=None, shape='simple'):
        self.g = 9.81
        self.m = m
        self.c = c
        self.L = L  # 对于单摆是摆长，对于复摆是等效摆长用于绘图

        # 确定转动惯量和重心距
        if shape == 'simple':
            self.d = L
            self.I = m * L ** 2
        else:
            # 假设复摆参数传入时，L代表重心距d，I_pivot需手动指定
            self.d = L
            self.I = I_pivot if I_pivot else (1 / 3) * m * L ** 2

            # 固有频率 (用于计算推荐的仿真时长)
        self.omega_n = np.sqrt(self.m * self.g * self.d / self.I)
        self.T_period = 2 * np.pi / self.omega_n

    def model_derivatives(self, y, t):
        theta, omega = y
        dtheta = omega
        domega = (-self.m * self.g * self.d * np.sin(theta) - self.c * omega) / self.I
        return [dtheta, domega]

    def run_simulation(self, theta0, periods=60, fps=60):
        t_max = periods * self.T_period
        t = np.linspace(0, t_max, int(t_max * fps))
        y0 = [theta0, 0.0]
        solution = odeint(self.model_derivatives, y0, t)
        return t, solution[:, 0], solution[:, 1]


class VideoRenderer:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.pivot = (width // 2, height // 4)

    def render(self, filename, t_data, theta_data, L_visual, fps):
        print(f"--- 正在渲染视频: {filename} ---")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (self.width, self.height))

        # 计算缩放比例: 让摆长占据屏幕合适高度
        scale = (self.height * 0.6) / L_visual

        for theta in tqdm(theta_data, desc="Writing Frames"):
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            # 计算球坐标
            x = int(self.pivot[0] + L_visual * np.sin(theta) * scale)
            y = int(self.pivot[1] + L_visual * np.cos(theta) * scale)

            # 绘图 (高对比度：白线白球，黑底)
            cv2.line(frame, self.pivot, (x, y), (200, 200, 200), 2)
            cv2.circle(frame, (x, y), 20, (255, 255, 255), -1)

            out.write(frame)

        out.release()
        print("视频保存成功。")
        return self.pivot, scale


if __name__ == "__main__":
    # --- 1. 配置参数 ---
    output_video = "pendulum_experiment.mp4"
    output_data = "ground_truth.npz"

    # 模拟一个有阻尼的单摆
    params = {
        'm': 1.0,  # 质量 kg
        'L': 1.0,  # 摆长 m
        'c': 0.05,  # 阻尼系数
        'shape': 'simple'
    }

    # --- 2. 运行物理仿真 ---
    sim = PendulumSimulator(**params)
    print(f"物理模型初始化: 固有周期 T ≈ {sim.T_period:.2f}s")

    # 仿真 60 个周期
    t, theta, omega = sim.run_simulation(theta0=np.radians(30), periods=60, fps=60)

    # --- 3. 保存真值数据 (用于后续对比验证) ---
    np.savez(output_data, t=t, theta=theta, omega=omega, params=params)
    print(f"真值数据已保存至 {output_data}")

    # --- 4. 生成视频 ---
    renderer = VideoRenderer()
    renderer.render(output_video, t, theta, params['L'], fps=60)