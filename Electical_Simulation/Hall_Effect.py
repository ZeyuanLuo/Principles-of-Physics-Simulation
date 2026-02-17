import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# --- 1. 物理引擎层 (模拟真实的物理现象) ---
class HallEffectSimulator:
    def __init__(self):
        # 物理常数 (模拟砷化镓 GaAs)
        self.n = 1.0e21  # 载流子浓度 (m^-3)
        self.d = 1.0e-3  # 元件厚度 (m)
        self.e = 1.602e-19  # 电子电荷
        # 模拟电磁铁系数: B = k * I_m
        self.k_magnet = 0.5  # T/A

    def get_hall_voltage(self, I_sample, I_magnet, add_noise=True):
        """
        计算霍尔电压 V_H = (I * B) / (n * e * d)
        I_sample: 流过霍尔元件的电流 (A)
        I_magnet: 励磁电流 (A)
        """
        B = self.k_magnet * I_magnet
        # 理论电压
        V_H = (I_sample * B) / (self.n * self.e * self.d) * 1000  # 转换为 mV

        # 模拟真实实验中的噪声和干扰
        if add_noise:
            noise = np.random.normal(0, 0.5)  # 高斯白噪声
            V_H += noise

        return V_H, B


# --- 2. AI 智能分析层 ---
def linear_model(x, a, b):
    return a * x + b


def ai_analyze_data(x_data, y_data):
    """模拟 AI Box 的自动拟合功能"""
    popt, pcov = curve_fit(linear_model, x_data, y_data)
    sensitivity = popt[0]  # 斜率即灵敏度
    r_squared = 1 - (np.sum((y_data - linear_model(x_data, *popt)) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2))
    return popt, r_squared


# --- 3. 交互界面层 (Streamlit) ---
st.set_page_config(page_title="AI 物理实验室 - 霍尔效应", layout="wide")

st.title("🔬 虚拟物理实验室：霍尔效应 (AI 增强版)")
st.markdown("---")

col1, col2 = st.columns([1, 2])

# 左侧：控制面板 (模拟硬件旋钮)
with col1:
    st.header("🎛️ 设备控制台")

    st.subheader("1. 霍尔元件参数")
    I_sample_mA = st.slider("工作电流 $I_S$ (mA)", 0.0, 3.5, 1.0, 0.1)

    st.subheader("2. 电磁铁电源")
    I_magnet_mA = st.slider("励磁电流 $I_M$ (mA)", 0, 1000, 500, 10)

    st.info("💡 提示：改变励磁电流 $I_M$ 来改变磁场强度 $B$。")

    # 模拟数据采集按钮
    run_experiment = st.button("🔴 开始采集一组数据 ($I_M$: 0-1000mA)")

# 右侧：实验现象与数据
with col2:
    sim = HallEffectSimulator()

    # 实时单点测量显示
    current_V_H, current_B = sim.get_hall_voltage(I_sample_mA / 1000, I_magnet_mA / 1000)

    # 模拟数字仪表
    st.header("📟 实时读数")
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("霍尔电压 $V_H$", f"{current_V_H:.2f} mV")
    m_col2.metric("磁感应强度 $B$", f"{current_B:.3f} T")
    m_col3.metric("理论误差", "±0.5 mV")

    # 如果点击了“开始采集”，模拟自动扫描过程
    if run_experiment:
        st.header("📈 AI 数据分析")

        # 1. 自动生成数据 (模拟 AI Box 控制电流扫描)
        scan_I_M = np.linspace(0, 1000, 20)  # 采集20个点
        scan_V_H = []
        for i_m in scan_I_M:
            v, _ = sim.get_hall_voltage(I_sample_mA / 1000, i_m / 1000)
            scan_V_H.append(v)

        scan_V_H = np.array(scan_V_H)

        # 2. AI 进行拟合
        popt, r2 = ai_analyze_data(scan_I_M, scan_V_H)
        fit_y = linear_model(scan_I_M, *popt)

        # 3. 绘图
        fig, ax = plt.subplots()
        ax.scatter(scan_I_M, scan_V_H, label='Measured Data (with Noise)', color='blue', s=10)
        ax.plot(scan_I_M, fit_y, label=f'AI Fit (R²={r2:.4f})', color='red', linestyle='--')
        ax.set_xlabel('Excitation Current $I_M$ (mA)')
        ax.set_ylabel('Hall Voltage $V_H$ (mV)')
        ax.set_title(f'Hall Voltage vs. Magnetic Field (at $I_S$={I_sample_mA}mA)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)

        # 4. AI 报告
        st.success(f"✅ AI 分析完成：\n\n"
                   f"- 霍尔灵敏度 $K_H$: {popt[0]:.4f} mV/mA\n"
                   f"- 线性度 (R²): {r2:.4f}\n"
                   f"- 载流子浓度估算: 自动计算中...")