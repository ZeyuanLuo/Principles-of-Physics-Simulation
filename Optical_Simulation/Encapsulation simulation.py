import sys
import cv2
import numpy as np
import time

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QLineEdit,
                             QComboBox, QGroupBox, QFormLayout)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap


# --- 核心算法类 (负责图像处理与计算) ---
class NewtonAnalyzer:
    def __init__(self):
        self.lam = 589.3 * 1e-9  # 默认钠光波长
        self.pix_size = 10 * 1e-6  # 默认像素尺寸

    def update_params(self, lam_nm, pix_um):
        self.lam = float(lam_nm) * 1e-9
        self.pix_size = float(pix_um) * 1e-6

    def process(self, img):
        """
        这里执行图像识别逻辑。
        为了演示流畅性，这里包含了一个简单的模拟识别展示。
        """
        if img is None: return img, 0, 0

        # 1. 转灰度
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        h, w = gray.shape
        center = (w // 2, h // 2)

        # 2. 图像处理演示：在图上画十字和圆圈
        display_img = img.copy()

        # 画十字准星
        cv2.drawMarker(display_img, center, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

        # 模拟：画几个绿色的圆环，假装识别到了暗纹
        # (实际项目中，这里替换为真实的 find_peaks 代码)
        for r in [50, 80, 110, 140]:
            cv2.circle(display_img, center, r, (0, 255, 0), 1)

        # 3. 模拟计算结果
        # 实际项目中，这里使用线性拟合 slope 计算 R
        # 这里为了演示，生成一个在 2500mm 附近微小波动的值
        noise = np.random.uniform(-5, 5)
        R_calc = 2500.0 + noise

        fit_quality = 0.9982  # 模拟拟合度

        return display_img, R_calc, fit_quality


# --- 视频采集线程 (后台运行，防止界面卡顿) ---
class VideoThread(QThread):
    # 定义信号：用于把数据传回主界面
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_result_signal = pyqtSignal(float, float)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.camera_id = 0
        self.analyzer = NewtonAnalyzer()
        self.enable_analysis = False

    def run(self):
        # 连接摄像头
        # 如果您有工业相机，通常在这里改为 cv2.VideoCapture(1) 或其他索引
        cap = cv2.VideoCapture(self.camera_id)

        # 尝试设置高分辨率 (取决于相机支持)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                if self.enable_analysis:
                    # 如果开启了识别，就跑算法
                    final_img, R, fit = self.analyzer.process(cv_img)
                    self.update_result_signal.emit(R, fit)
                else:
                    # 没开启识别，就只显示原图
                    final_img = cv_img

                # 发送图像信号给界面
                self.change_pixmap_signal.emit(final_img)
            else:
                # 读不到图像时稍微休息下，避免死循环占满CPU
                self.msleep(100)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


# --- 主界面 ---
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("牛顿环智能分析系统 (PyQt5版)")
        self.resize(1000, 700)

        # 设置中心窗口
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout()
        self.central_widget.setLayout(self.main_layout)

        # === 界面左侧：视频显示区域 ===
        self.image_label = QLabel("摄像头未启动")
        # 【修改点1】PyQt5 写法：Qt.AlignCenter (没有 AlignmentFlag)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #2b2b2b; color: white; font-size: 20px;")
        self.image_label.setMinimumSize(640, 480)
        self.main_layout.addWidget(self.image_label, stretch=3)

        # === 界面右侧：控制面板 ===
        self.controls_layout = QVBoxLayout()
        self.main_layout.addLayout(self.controls_layout, stretch=1)

        self.setup_controls()

        # 初始化后台线程
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_result_signal.connect(self.update_data)

    def setup_controls(self):
        # -- 1. 硬件设置 --
        grp_cam = QGroupBox("硬件连接")
        layout_cam = QFormLayout()

        self.combo_cam = QComboBox()
        self.combo_cam.addItems(["摄像头 0 (默认)", "摄像头 1 (外接)", "摄像头 2"])

        self.btn_start = QPushButton("启动相机")
        self.btn_start.clicked.connect(self.toggle_camera)
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")

        layout_cam.addRow("选择设备:", self.combo_cam)
        layout_cam.addRow(self.btn_start)
        grp_cam.setLayout(layout_cam)
        self.controls_layout.addWidget(grp_cam)

        # -- 2. 实验参数 --
        grp_param = QGroupBox("实验参数输入")
        layout_param = QFormLayout()

        self.input_wave = QLineEdit("589.3")  # 默认钠光
        self.input_pix = QLineEdit("10.0")  # 默认像素大小
        self.btn_apply = QPushButton("更新参数")
        self.btn_apply.clicked.connect(self.apply_params)

        layout_param.addRow("光波长 (nm):", self.input_wave)
        layout_param.addRow("像素尺寸 (um):", self.input_pix)
        layout_param.addRow(self.btn_apply)
        grp_param.setLayout(layout_param)
        self.controls_layout.addWidget(grp_param)

        # -- 3. 结果显示 --
        grp_res = QGroupBox("实时测量数据")
        layout_res = QVBoxLayout()

        self.lbl_R = QLabel("R = 0.00 mm")
        self.lbl_R.setStyleSheet("font-size: 24px; color: blue; font-weight: bold;")

        self.lbl_Fit = QLabel("线性度 (R²) = 0.000")
        self.lbl_Fit.setStyleSheet("font-size: 14px; color: #333;")

        self.btn_toggle_algo = QPushButton("开始自动测量")
        self.btn_toggle_algo.setCheckable(True)
        self.btn_toggle_algo.setStyleSheet("padding: 8px;")
        self.btn_toggle_algo.clicked.connect(self.toggle_algo)

        layout_res.addWidget(QLabel("曲率半径结果:"))
        layout_res.addWidget(self.lbl_R)
        layout_res.addWidget(self.lbl_Fit)
        layout_res.addStretch()
        layout_res.addWidget(self.btn_toggle_algo)
        grp_res.setLayout(layout_res)
        self.controls_layout.addWidget(grp_res)

        self.controls_layout.addStretch()

    def toggle_camera(self):
        if self.thread.isRunning():
            self.thread.stop()
            self.btn_start.setText("启动相机")
            self.image_label.setText("摄像头已关闭")
            self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        else:
            self.thread.camera_id = self.combo_cam.currentIndex()
            self.thread.start()
            self.btn_start.setText("停止相机")
            self.btn_start.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 10px;")

    def toggle_algo(self, checked):
        self.thread.enable_analysis = checked
        if checked:
            self.btn_toggle_algo.setText("停止测量")
            self.btn_toggle_algo.setStyleSheet("background-color: #f44336; color: white; padding: 8px;")
        else:
            self.btn_toggle_algo.setText("开始自动测量")
            self.btn_toggle_algo.setStyleSheet("padding: 8px;")

    def apply_params(self):
        try:
            w = float(self.input_wave.text())
            p = float(self.input_pix.text())
            self.thread.analyzer.update_params(w, p)
            print(f"参数已更新: 波长={w}nm, 像素={p}um")
        except ValueError:
            print("请输入有效的数字！")

    def update_image(self, cv_img):
        """将OpenCV图像转换为Qt图像并显示"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w

        # 【修改点2】PyQt5 写法：QImage.Format_RGB888 (没有 .Format)
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 【修改点3】PyQt5 写法：Qt.KeepAspectRatio (没有 AspectRatioMode)
        p = convert_to_Qt_format.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)

        self.image_label.setPixmap(QPixmap.fromImage(p))

    def update_data(self, R, fit):
        self.lbl_R.setText(f"R = {R:.2f} mm")
        self.lbl_Fit.setText(f"线性度 (R²) = {fit:.4f}")

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    # 【修改点4】PyQt5 推荐写法：app.exec_() (多一个下划线)
    sys.exit(app.exec_())