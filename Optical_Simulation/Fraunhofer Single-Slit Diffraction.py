import cv2
import numpy as np
import matplotlib.pyplot as plt
import io


class DiffractionSimulator:
    """
    夫琅禾费单缝衍射模拟器
    """

    def __init__(self, width=800, height=400, a_um=100, D_m=1.0, lambda_nm=532):
        self.w = width
        self.h = height
        self.a = a_um * 1e-6  # 缝宽 (米)
        self.D = D_m  # 屏距 (米)
        self.lam = lambda_nm * 1e-9  # 波长 (米)
        self.pixel_size = 10e-6  # 假设相机像素尺寸 10微米

    def generate_image(self):
        # 1. 创建坐标系
        # 这是一个 1D 衍射扩展到 2D 图像
        x_indices = np.linspace(-self.w // 2, self.w // 2, self.w)
        x_meters = x_indices * self.pixel_size

        # 2. 计算 Beta
        # sin(theta) approx x / D
        sin_theta = x_meters / self.D
        beta = (np.pi * self.a * sin_theta) / self.lam

        # 3. 计算光强 I = I0 * (sin(beta)/beta)^2
        # 注意: numpy.sinc(x) 定义为 sin(pi*x)/(pi*x)，所以我们这里需要调整输入
        # 物理公式中的 beta 包含了 pi，所以传给 np.sinc 的应该是 (a * sin_theta / lambda)
        sinc_input = (self.a * sin_theta) / self.lam
        intensity = np.sinc(sinc_input) ** 2

        # 4. 生成图像
        # 归一化并扩展到二维 (垂直方向重复)
        img_1d = (intensity * 255).astype(np.uint8)
        img_2d = np.tile(img_1d, (self.h, 1))

        # 5. 添加噪声和过曝效果 (模拟真实拍摄)
        noise = np.random.normal(0, 2, img_2d.shape)
        img_noisy = np.clip(img_2d + noise, 0, 255).astype(np.uint8)

        return img_noisy, intensity


class DiffractionAnalyzer:
    """
    衍射条纹分析器
    """

    def __init__(self, D_m, lambda_nm, pixel_size_um):
        self.D = D_m
        self.lam = lambda_nm * 1e-9
        self.pixel_size = pixel_size_um * 1e-6

    def analyze(self, img):
        h, w = img.shape
        center_y = h // 2
        center_x = w // 2

        # 1. 获取光强分布曲线 (取中间一行的平均值以减少噪声)
        # 取中间 50 行的平均
        roi = img[center_y - 25:center_y + 25, :]
        profile = np.mean(roi, axis=0)

        # 2. 寻找暗纹 (Local Minima)
        # 也就是寻找 profile 的波谷
        # 为了稳健，我们先平滑一下
        profile_smooth = cv2.GaussianBlur(profile.reshape(1, -1), (11, 11), 0).flatten()

        minima_indices = []
        # 从中心向两侧搜索
        # 为了避免把图像边缘当成暗纹，留出边距
        for i in range(20, w - 20):
            # 简单的波谷检测: 比左右都小，且绝对亮度较低
            if profile_smooth[i] < profile_smooth[i - 1] and \
                    profile_smooth[i] < profile_smooth[i + 1]:

                # 过滤掉亮度太高的地方 (那是亮纹及其起伏)
                # 过滤掉中心附近的波动
                if profile_smooth[i] < 50:
                    minima_indices.append(i)

        # 3. 数据配对 (确定级数 k)
        # 暗纹级数 k = +/-1, +/-2 ...
        # 我们根据距离中心的远近来分配 k
        results_k = []  # 级数
        results_x = []  # 距离中心的像素距离
        results_idx = []  # 原始索引

        for idx in minima_indices:
            dist = idx - center_x
            if abs(dist) < 10: continue  # 忽略正中心可能误判的点

            # 这里的 k 需要根据顺序推断，简单的方法是按距离排序
            # 暂时只记录绝对距离
            results_x.append(dist * self.pixel_size)  # 转换为米
            results_idx.append(idx)

        # 简单排序并分配 k 值 (假设对称)
        # 这里做一个简单的聚类假设：越靠近中心的也是越小的 k
        # 为了演示简单，我们通过拟合所有点 x = k * (D*lambda/a)
        # 我们可以根据 x 的正负来分配 k 的正负

        # 智能分配 k 值：
        # 理论间距 delta_x = D * lambda / a_est
        # 我们可以先粗略估计一个 delta_x，或者直接对 x 排序
        # 简单方案：将 x 也就是距离，除以一个估计的最小间隔，取整

        final_k = []
        final_x = []
        valid_indices = []

        if len(results_x) > 0:
            abs_x = np.abs(results_x)
            min_dist = np.min(abs_x)  # 第一级暗纹的大致距离

            for i, x_val in enumerate(results_x):
                # k ≈ x / x_1
                k_est = round(x_val / min_dist)
                if k_est != 0:
                    final_k.append(k_est)
                    final_x.append(x_val)
                    valid_indices.append(results_idx[i])

        return final_k, final_x, valid_indices, profile

    def calculate_slit_width(self, k_list, x_list):
        if len(k_list) < 2: return 0, 0, 0, 0

        k_arr = np.array(k_list)
        x_arr = np.array(x_list)

        # 物理公式: x = k * (D * lambda / a)
        # 也就是 x = Slope * k
        # 其中 Slope = (D * lambda) / a
        # 所以 a = (D * lambda) / Slope

        # 线性拟合 x = m * k (过原点，也可以不过，这里允许微小截距)
        slope, intercept = np.polyfit(k_arr, x_arr, 1)

        # 计算缝宽 a
        if slope != 0:
            a_calc = (self.D * self.lam) / slope
        else:
            a_calc = 0

        # 拟合质量 R^2
        correlation = np.corrcoef(k_arr, x_arr)[0, 1]
        r_sq = correlation ** 2

        return a_calc, r_sq, slope, intercept


class ReportDrawer:
    def draw_report(self, img, profile, valid_indices, k_list, x_list, a_calc, r_sq, slope, intercept, true_a):
        h, w = img.shape

        # 1. 准备画布
        canvas = np.ones((700, 1000, 3), dtype=np.uint8) * 240

        # 2. 左上：衍射原图
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # 标记暗纹位置
        for idx in valid_indices:
            cv2.line(img_color, (idx, 0), (idx, h), (0, 255, 0), 1)  # 绿色竖线

        # 缩放放入画布
        target_w = 600
        scale = target_w / w
        img_disp = cv2.resize(img_color, (0, 0), fx=scale, fy=scale)
        dh, dw, _ = img_disp.shape
        canvas[20:20 + dh, 20:20 + dw] = img_disp

        cv2.putText(canvas, "Captured Diffraction Pattern", (20, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 3. 左下：光强分布曲线 (Profile)
        # 使用 Matplotlib 绘制并转为 OpenCV
        fig1, ax1 = plt.subplots(figsize=(6, 3), dpi=100)
        ax1.plot(profile, 'k-', linewidth=1, label='Intensity')
        ax1.plot(valid_indices, [profile[i] for i in valid_indices], 'ro', label='Minima')
        ax1.set_title("Intensity Profile")
        ax1.set_xlabel("Pixel Position")
        ax1.set_ylabel("Gray Value")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()

        buf1 = io.BytesIO()
        plt.savefig(buf1, format='png')
        buf1.seek(0)
        plot1 = cv2.imdecode(np.frombuffer(buf1.getvalue(), dtype=np.uint8), 1)
        plt.close(fig1)

        # 放入画布
        ph1, pw1, _ = plot1.shape
        y_offset = 20 + dh + 30
        canvas[y_offset:y_offset + ph1, 20:20 + pw1] = plot1

        # 4. 右侧：线性拟合图 (x vs k)
        fig2, ax2 = plt.subplots(figsize=(3.5, 4), dpi=100)
        ax2.scatter(k_list, np.array(x_list) * 1000, color='blue', label='Data')  # x转换为mm

        k_fit = np.linspace(min(k_list) - 0.5, max(k_list) + 0.5, 100)
        x_fit = slope * k_fit + intercept
        ax2.plot(k_fit, x_fit * 1000, 'r--', label='Fit')

        ax2.set_title(r"Position $x_k$ vs Order $k$")
        ax2.set_xlabel("Order (k)")
        ax2.set_ylabel("Position x (mm)")
        ax2.grid(True, linestyle='--')
        plt.tight_layout()

        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png')
        buf2.seek(0)
        plot2 = cv2.imdecode(np.frombuffer(buf2.getvalue(), dtype=np.uint8), 1)
        plt.close(fig2)

        # 放入画布
        ph2, pw2, _ = plot2.shape
        x_right = 20 + dw + 20
        canvas[20:20 + ph2, x_right:x_right + pw2] = plot2

        # 5. 右下：文字结果
        text_x = x_right
        text_y = 20 + ph2 + 40

        cv2.putText(canvas, "Analysis Result", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        text_y += 30

        # 显示误差
        error = abs(a_calc - true_a) / true_a * 100

        lines = [
            f"True Slit Width: {true_a * 1e6:.1f} um",
            f"Calc Slit Width: {a_calc * 1e6:.1f} um",
            f"Error: {error:.2f}%",
            f"Linearity (R2): {r_sq:.4f}",
            "-" * 20,
            "Formula:",
            "a * sin(theta) = k * lambda",
            "x = k * (D * lambda / a)"
        ]

        for i, line in enumerate(lines):
            color = (0, 0, 0)
            if "Error" in line:
                color = (0, 0, 255) if error > 2 else (0, 100, 0)
            elif "Calc" in line:
                color = (255, 0, 0)

            cv2.putText(canvas, line, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            text_y += 25

        return canvas


def main():
    # --- 实验参数设置 ---
    SLIT_WIDTH_UM = 150.0  # 真实缝宽 150微米
    DISTANCE_M = 1.0  # 屏距 1米
    WAVELENGTH_NM = 632.8  # 红色激光
    PIX_SIZE_UM = 10.0  # 像素大小

    print(f"开始模拟夫琅禾费单缝衍射 (a={SLIT_WIDTH_UM}um)...")

    # 1. 模拟
    sim = DiffractionSimulator(width=1280, height=300,
                               a_um=SLIT_WIDTH_UM,
                               D_m=DISTANCE_M,
                               lambda_nm=WAVELENGTH_NM)
    img, intensity_data = sim.generate_image()

    # 2. 识别与分析
    print("正在识别暗纹并计算缝宽...")
    analyzer = DiffractionAnalyzer(D_m=DISTANCE_M,
                                   lambda_nm=WAVELENGTH_NM,
                                   pixel_size_um=PIX_SIZE_UM)

    k_list, x_list, valid_indices, profile = analyzer.analyze(img)
    a_calc, r_sq, slope, intercept = analyzer.calculate_slit_width(k_list, x_list)

    # 3. 绘图报告
    print("正在生成报告...")
    drawer = ReportDrawer()
    report_img = drawer.draw_report(img, profile, valid_indices, k_list, x_list,
                                    a_calc, r_sq, slope, intercept, SLIT_WIDTH_UM * 1e-6)

    # 4. 输出
    print(f"真实缝宽: {SLIT_WIDTH_UM} um")
    print(f"计算缝宽: {a_calc * 1e6:.2f} um")

    cv2.imshow("Fraunhofer Diffraction Analysis", report_img)
    cv2.imwrite("diffraction_report.png", report_img)
    print("结果已保存为 diffraction_report.png")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()