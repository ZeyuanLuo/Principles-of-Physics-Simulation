import sys
import cv2
import numpy as np
import time
from datetime import datetime

# PyQt5 依赖
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QGroupBox,
                             QLCDNumber, QMessageBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap


# --- 核心算法类：计数 + 箭头绘制 ---
class MichelsonCore:
    def __init__(self):
        self.prev_gray = None
        self.prev_intensity = 0
        self.total_count = 0
        self.direction = 0

        # 参数
        self.intensity_threshold = 120
        self.flow_threshold = 0.05

        # 网格参数 (用于光流)
        self.h, self.w = 0, 0
        self.unit_radial_x = None
        self.unit_radial_y = None
        self.grid_step = 30  # 箭头稀疏度，越小箭头越密

    def init_grid(self, shape):
        self.h, self.w = shape
        y, x = np.mgrid[0:self.h, 0:self.w]
        center_x, center_y = self.w // 2, self.h // 2

        # 径向单位向量
        fx = x - center_x
        fy = y - center_y
        norm = np.sqrt(fx ** 2 + fy ** 2)
        norm[norm == 0] = 1
        self.unit_radial_x = fx / norm
        self.unit_radial_y = fy / norm

        # 箭头绘制用的网格采样点
        self.y_grid, self.x_grid = np.mgrid[self.grid_step // 2:self.h:self.grid_step,
                                   self.grid_step // 2:self.w:self.grid_step].reshape(2, -1).astype(int)

    def process(self, frame):
        if frame is None: return frame, 0, 0

        # 1. 预处理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 初始化
        if self.unit_radial_x is None or gray.shape != (self.h, self.w):
            self.init_grid(gray.shape)
            self.prev_gray = gray
            self.prev_intensity = np.mean(gray[self.h // 2 - 5:self.h // 2 + 5, self.w // 2 - 5:self.w // 2 + 5])
            return frame, 0, 0

        # 2. 光流计算
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        radial_flow = flow[..., 0] * self.unit_radial_x + flow[..., 1] * self.unit_radial_y
        mean_velocity = np.mean(radial_flow)

        # 3. 状态更新
        if mean_velocity > self.flow_threshold:
            self.direction = 1
        elif mean_velocity < -self.flow_threshold:
            self.direction = -1
        else:
            self.direction = 0

        # 4. 计数逻辑 (亮度触发 + 方向门控)
        cy, cx = self.h // 2, self.w // 2
        roi = gray[cy - 5:cy + 5, cx - 5:cx + 5]
        curr_intensity = np.mean(roi)

        if self.prev_intensity < self.intensity_threshold and curr_intensity >= self.intensity_threshold:
            if self.direction != 0:
                self.total_count += self.direction

        self.prev_gray = gray
        self.prev_intensity = curr_intensity

        # 5. 绘图 (把箭头画回去！)
        display_img = frame.copy()

        # A. 绘制光流箭头 (您要求保留的直观功能)
        if self.direction != 0:  # 只有动的时候才画，防止乱闪
            fx, fy = flow[self.y_grid, self.x_grid].T
            # 放大箭头长度以便观察
            lines = np.vstack([self.x_grid, self.y_grid,
                               self.x_grid + fx * 8, self.y_grid + fy * 8]).T.reshape(-1, 2, 2)
            lines = np.int32(lines + 0.5)

            # 涌出用绿色箭头，塌陷用红色箭头
            arrow_color = (0, 255, 0) if self.direction > 0 else (0, 0, 255)
            for (x1, y1), (x2, y2) in lines:
                # 只有当该点局部速度够大才画，过滤噪点
                dist = (x1 - x2) ** 2 + (y1 - y2) ** 2
                if dist > 2:
                    cv2.arrowedLine(display_img, (x1, y1), (x2, y2), arrow_color, 1, tipLength=0.3)

        # B. 绘制中心框和文字
        color_box = (0, 255, 0) if curr_intensity > self.intensity_threshold else (0, 0, 255)
        cv2.rectangle(display_img, (cx - 8, cy - 8), (cx + 8, cy + 8), color_box, 2)

        status_text = "Static"
        if self.direction == 1:
            status_text = "Emerging (+)"
        elif self.direction == -1:
            status_text = "Vanishing (-)"

        cv2.putText(display_img, f"Status: {status_text}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return display_img, self.total_count, mean_velocity


# --- 模拟器 ---
class MichelsonSimulator:
    def __init__(self, width=640, height=480):
        self.width, self.height = width, height
        x = np.linspace(-10, 10, width)
        y = np.linspace(-10, 10, height)
        self.X, self.Y = np.meshgrid(x, y)
        self.R2 = self.X ** 2 + self.Y ** 2
        self.phase = 0.0
        self.dir = 1

    def get_frame(self):
        self.phase += 0.2 * self.dir
        if self.phase > 15: self.dir = -1
        if self.phase < -5: self.dir = 1
        intensity = np.cos(0.5 * self.R2 - self.phase) ** 2
        frame = (intensity * 255).astype(np.uint8)
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


# --- 视频线程 (集成录像功能) ---
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_data_signal = pyqtSignal(int, float)
    recording_status_signal = pyqtSignal(bool, str)  # 反馈录制状态

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.use_camera = False
        self.camera_id = 0
        self.core = MichelsonCore()
        self.simulator = MichelsonSimulator()

        # 录像相关
        self.is_recording = False
        self.video_writer = None

    def start_recording(self):
        if self.is_recording: return

        # 生成文件名: "Record_20231027_153022.mp4"
        filename = f"Record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

        # 获取当前分辨率 (必须与 frame 一致)
        w = self.core.w if self.core.w > 0 else 640
        h = self.core.h if self.core.h > 0 else 480

        # 初始化 writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))

        if self.video_writer.isOpened():
            self.is_recording = True
            self.recording_status_signal.emit(True, filename)
        else:
            self.recording_status_signal.emit(False, "无法创建文件")

    def stop_recording(self):
        self.is_recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.recording_status_signal.emit(False, "Stopped")

    def run(self):
        cap = None
        if self.use_camera:
            cap = cv2.VideoCapture(self.camera_id)
            # 尝试设置高分辨率
            cap.set(3, 1280)
            cap.set(4, 720)

        while self._run_flag:
            if self.use_camera:
                ret, frame = cap.read()
                if not ret:
                    self.msleep(100)
                    continue
            else:
                frame = self.simulator.get_frame()
                self.msleep(50)  # 模拟帧率

            # 1. 算法处理 (会画箭头)
            final_img, count, vel = self.core.process(frame)

            # 2. 如果正在录制，写入视频流
            if self.is_recording and self.video_writer:
                self.video_writer.write(final_img)

            # 3. 发送给界面显示
            self.change_pixmap_signal.emit(final_img)
            self.update_data_signal.emit(count, vel)

        # 退出清理
        if cap: cap.release()
        if self.video_writer: self.video_writer.release()

    def stop(self):
        self._run_flag = False
        self.stop_recording()  # 确保保存
        self.wait()

    def reset_count(self):
        self.core.total_count = 0


# --- 主界面 ---
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("迈克尔逊全功能分析仪 (计数+箭头+录像)")
        self.resize(1000, 650)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout()
        main_widget.setLayout(layout)

        # 1. 左侧显示区
        self.image_label = QLabel("正在初始化...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #1e1e1e; color: gray; border: 2px solid #333;")
        self.image_label.setMinimumSize(640, 480)
        layout.addWidget(self.image_label, stretch=3)

        # 2. 右侧控制区
        ctrl_panel = QVBoxLayout()
        layout.addLayout(ctrl_panel, stretch=1)

        # 2.1 数据仪表盘
        self.lcd = QLCDNumber()
        self.lcd.setDigitCount(4)
        self.lcd.setStyleSheet("background: black; color: #00ff00; border: 2px solid gray;")
        self.lcd.setMinimumHeight(80)
        ctrl_panel.addWidget(QLabel("<h2>条纹计数 (Count)</h2>"))
        ctrl_panel.addWidget(self.lcd)

        self.lbl_vel = QLabel("流速: 0.00")
        self.lbl_vel.setStyleSheet("font-size: 14pt; margin-top: 10px;")
        ctrl_panel.addWidget(self.lbl_vel)

        # 2.2 录像控制区 (新增)
        grp_rec = QGroupBox("视频录制")
        grp_rec.setStyleSheet("QGroupBox { font-weight: bold; }")
        layout_rec = QVBoxLayout()

        self.btn_record = QPushButton("● 开始录制")
        self.btn_record.setStyleSheet("color: white; background-color: #d32f2f; font-weight: bold; padding: 10px;")
        self.btn_record.clicked.connect(self.toggle_recording)

        self.lbl_rec_status = QLabel("状态: 待机")
        self.lbl_rec_status.setStyleSheet("color: gray;")

        layout_rec.addWidget(self.btn_record)
        layout_rec.addWidget(self.lbl_rec_status)
        grp_rec.setLayout(layout_rec)
        ctrl_panel.addWidget(grp_rec)

        # 2.3 常规控制
        grp_ctrl = QGroupBox("系统控制")
        layout_ctrl = QVBoxLayout()

        self.btn_reset = QPushButton("重置计数")
        self.btn_reset.clicked.connect(self.reset_cnt)

        self.btn_source = QPushButton("切换: 模拟 / 相机")
        self.btn_source.setCheckable(True)
        self.btn_source.clicked.connect(self.switch_source)

        layout_ctrl.addWidget(self.btn_reset)
        layout_ctrl.addWidget(self.btn_source)
        grp_ctrl.setLayout(layout_ctrl)
        ctrl_panel.addWidget(grp_ctrl)

        ctrl_panel.addStretch()

        # 线程启动
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_data_signal.connect(self.update_dash)
        self.thread.recording_status_signal.connect(self.on_rec_status)
        self.thread.start()

    def update_image(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))

    def update_dash(self, count, vel):
        self.lcd.display(count)
        self.lbl_vel.setText(f"流速: {vel:.4f}")
        # 根据速度变色
        if vel > 0.05:
            self.lbl_vel.setStyleSheet("font-size: 14pt; color: green; font-weight: bold;")
        elif vel < -0.05:
            self.lbl_vel.setStyleSheet("font-size: 14pt; color: red; font-weight: bold;")
        else:
            self.lbl_vel.setStyleSheet("font-size: 14pt; color: black;")

    def toggle_recording(self):
        if not self.thread.is_recording:
            self.thread.start_recording()
        else:
            self.thread.stop_recording()

    def on_rec_status(self, is_rec, msg):
        if is_rec:
            self.btn_record.setText("■ 停止录制")
            self.btn_record.setStyleSheet("background-color: #333; color: white; padding: 10px;")
            self.lbl_rec_status.setText(f"正在录制: {msg}")
            self.lbl_rec_status.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.btn_record.setText("● 开始录制")
            self.btn_record.setStyleSheet("background-color: #d32f2f; color: white; font-weight: bold; padding: 10px;")
            if msg == "Stopped":
                QMessageBox.information(self, "录制完成", "视频已保存到程序运行目录。")
                self.lbl_rec_status.setText("状态: 待机")
            else:
                self.lbl_rec_status.setText(msg)
            self.lbl_rec_status.setStyleSheet("color: gray;")

    def reset_cnt(self):
        self.thread.reset_count()

    def switch_source(self, checked):
        self.thread.stop()
        self.thread = VideoThread()
        self.thread.use_camera = checked
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_data_signal.connect(self.update_dash)
        self.thread.recording_status_signal.connect(self.on_rec_status)
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())