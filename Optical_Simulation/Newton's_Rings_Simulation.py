import cv2
import numpy as np
import matplotlib.pyplot as plt
import io


class NewtonRingSimulator:
    """
    物理模拟器：生成牛顿环干涉图像
    """

    def __init__(self, width=600, height=600, R_mm=2500, lambda_nm=589.3, pixel_size_um=15):
        self.w, self.h = width, height
        self.R = R_mm * 1e-3  # 换算为米
        self.lam = lambda_nm * 1e-9  # 换算为米
        self.pix_size = pixel_size_um * 1e-6  # 换算为米

        # 生成网格
        y, x = np.mgrid[-height // 2:height // 2, -width // 2:width // 2]
        self.physical_r2 = (x ** 2 + y ** 2) * (self.pix_size ** 2)

    def generate_image(self):
        # 干涉公式
        phase = np.pi * self.physical_r2 / (self.R * self.lam)
        intensity = np.sin(phase) ** 2
        img = (intensity * 255).astype(np.uint8)

        # 添加噪声使仿真更真实
        noise = np.random.normal(0, 3, img.shape).astype(np.int16)
        return np.clip(img + noise, 0, 255).astype(np.uint8)


class NewtonAnalyzer:
    """
    分析器：识别暗环并计算曲率半径
    """

    def __init__(self, lambda_nm, pixel_size_um):
        self.lam = lambda_nm * 1e-9
        self.pix_size = pixel_size_um * 1e-6

    def analyze(self, img):
        h, w = img.shape
        center = (w // 2, h // 2)

        # 1. 提取光强分布 (取横纵切片的平均值)
        line_h = img[center[1], center[0]:]
        line_v = img[center[1]:, center[0]]
        profile = (line_h.astype(float) + line_v.astype(float)) / 2.0

        # 高斯平滑
        profile = cv2.GaussianBlur(profile.reshape(1, -1), (5, 5), 0).flatten()

        # 2. 寻峰 (找波谷-暗纹)
        minima_indices = []
        # 从第15个像素开始找，避开中心的一大块黑斑
        for i in range(15, len(profile) - 10):
            if profile[i] < profile[i - 1] and profile[i] < profile[i + 1]:
                if profile[i] < 120:  # 阈值
                    # 防止相邻像素重复检测
                    if len(minima_indices) == 0 or (i - minima_indices[-1] > 8):
                        minima_indices.append(i)

        # 取前 10 个环
        minima_indices = minima_indices[:10]

        k_list = np.arange(1, len(minima_indices) + 1)
        r_pixel_list = np.array(minima_indices)

        return k_list, r_pixel_list, center

    def fit_data(self, k_list, r_pixel_list):
        if len(k_list) < 2: return 0, 0, 0, 0, []

        r_meters = r_pixel_list * self.pix_size
        r_sq = r_meters ** 2

        # --- 核心计算 ---
        # 线性拟合 r^2 = A * k + B
        # slope (斜率) = R * lambda
        # intercept (截距) = 可能存在的系统误差或级数偏移
        slope, intercept = np.polyfit(k_list, r_sq, 1)

        # 计算 R = 斜率 / 波长
        R_calc = slope / self.lam

        # 计算线性度 (R^2)
        correlation = np.corrcoef(k_list, r_sq)[0, 1]
        fit_quality = correlation ** 2

        return R_calc, fit_quality, slope, intercept, r_sq


class ReportGenerator:
    """
    报告生成器：绘制图表并合成大图
    """

    def __init__(self, true_R, wavelength, pix_size):
        self.true_R = true_R
        self.wavelength = wavelength
        self.pix_size = pix_size

    def create_plot_image(self, k_list, r_sq, slope, intercept):
        """ 使用 Matplotlib 绘制带截距的拟合线 """
        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)

        # 绘制数据点 (蓝色)
        # 将单位转为 mm^2 方便观察
        ax.scatter(k_list, r_sq * 1e6, color='blue', label='Data Points')

        # 绘制拟合线 (红色虚线)
        # 这里的关键修复：y = slope * x + intercept
        x_fit = np.linspace(0, max(k_list) + 1, 100)
        y_fit = (slope * x_fit + intercept) * 1e6

        intercept_val = intercept * 1e6
        sign_str = "+" if intercept_val >= 0 else "-"

        ax.plot(x_fit, y_fit, 'r--', label=f'Fit: y={slope * 1e6:.2e}x {sign_str} {abs(intercept_val):.2f}')

        ax.set_title(r'$r^2 - k$ Linear Regression')
        ax.set_xlabel('Ring Order (k)')
        ax.set_ylabel(r'$r^2$ ($mm^2$)')
        ax.legend(loc='upper left', fontsize='small')
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        # 保存到内存 Buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # 转为 OpenCV 图像
        plot_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        plot_img = cv2.imdecode(plot_arr, 1)
        plt.close(fig)
        return plot_img

    def generate_report(self, original_img, k_list, r_pixels, r_sq, center, R_calc, fit_quality, slope, intercept):
        h, w = original_img.shape

        # 1. 创建画布
        canvas_h = max(h, 600)
        canvas_w = w + 500
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 245  # 浅灰背景

        # 2. 左侧：绘制干涉图和标记
        display_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
        cv2.drawMarker(display_img, center, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

        for i, r in enumerate(r_pixels):
            cv2.circle(display_img, center, int(r), (0, 255, 0), 1)
            # 隔行标注级数，防止字挤在一起
            if i % 2 == 0:
                cv2.putText(display_img, str(k_list[i]), (center[0] + int(r) + 2, center[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 200), 1)

        canvas[0:h, 0:w] = display_img

        # 3. 右上：插入 Matplotlib 图表
        plot_img = self.create_plot_image(k_list, r_sq, slope, intercept)

        ph, pw, _ = plot_img.shape
        target_w = 480
        scale = target_w / pw
        plot_img_resized = cv2.resize(plot_img, (0, 0), fx=scale, fy=scale)
        ph_new, pw_new, _ = plot_img_resized.shape

        y_offset = 20
        x_offset = w + 10
        canvas[y_offset:y_offset + ph_new, x_offset:x_offset + pw_new] = plot_img_resized

        # 4. 右下：文字报告
        text_y = y_offset + ph_new + 40
        x_text = w + 20

        cv2.putText(canvas, "Measurement Report", (x_text, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        text_y += 40

        params = [
            f"Wavelength: {self.wavelength} nm",
            f"Pixel Size: {self.pix_size} um",
            f"True Radius: {self.true_R} mm"
        ]
        for p in params:
            cv2.putText(canvas, p, (x_text, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
            text_y += 25

        cv2.line(canvas, (x_text, text_y), (canvas_w - 20, text_y), (100, 100, 100), 1)
        text_y += 30

        error = abs(R_calc * 1000 - self.true_R) / self.true_R * 100

        # 结果高亮显示
        results = [
            (f"Calculated R: {R_calc * 1000:.2f} mm", (0, 100, 0)),
            (f"Linearity (R^2): {fit_quality:.5f}", (0, 0, 0)),
            (f"Error Rate: {error:.2f}%", (0, 0, 255) if error > 1 else (0, 100, 0))
        ]

        for text, color in results:
            cv2.putText(canvas, text, (x_text, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            text_y += 35

        # 简易数据表
        text_y += 20
        cv2.putText(canvas, "  k   |   r (px)  |   r^2 (mm^2)", (x_text, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1)
        text_y += 10
        cv2.line(canvas, (x_text, text_y), (canvas_w - 20, text_y), (200, 200, 200), 1)
        text_y += 20

        for k, r, r2 in zip(k_list[:5], r_pixels[:5], r_sq[:5]):
            row_str = f"  {k:<3} |   {r:<6.1f}  |   {r2 * 1e6:.4f}"
            cv2.putText(canvas, row_str, (x_text, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
            text_y += 20

        return canvas


def main():
    # --- 实验参数 ---
    R_TRUE = 3000  # 真实曲率半径 3000 mm
    WAVELENGTH = 589.3  # 钠光 589.3 nm
    PIX_SIZE = 12.0  # 像素尺寸 12 um

    # 1. 模拟生成图像
    print("正在模拟牛顿环...")
    sim = NewtonRingSimulator(R_mm=R_TRUE, lambda_nm=WAVELENGTH, pixel_size_um=PIX_SIZE)
    img = sim.generate_image()

    # 2. 图像识别与分析
    print("正在分析数据...")
    analyzer = NewtonAnalyzer(WAVELENGTH, PIX_SIZE)
    k_list, r_pixels, center = analyzer.analyze(img)

    # 获取拟合参数 (R, R^2, 斜率, 截距, r^2数组)
    R_calc, fit_q, slope, intercept, r_sq = analyzer.fit_data(k_list, r_pixels)

    # 3. 生成综合报告图
    print("正在生成可视化报告...")
    reporter = ReportGenerator(R_TRUE, WAVELENGTH, PIX_SIZE)
    # 【注意】这里传入了 intercept 参数
    report_img = reporter.generate_report(img, k_list, r_pixels, r_sq, center, R_calc, fit_q, slope, intercept)

    # 4. 显示与保存
    print(f"--- 结果 ---\n计算 R: {R_calc * 1000:.2f} mm\n真实 R: {R_TRUE} mm")

    filename = "newton_ring_report.png"
    cv2.imwrite(filename, report_img)
    print(f"报告已保存至: {filename}")

    cv2.imshow("Newton Rings Analysis Report (Fixed)", report_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()