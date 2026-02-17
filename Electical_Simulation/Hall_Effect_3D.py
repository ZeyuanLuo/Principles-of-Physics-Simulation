import streamlit as st
import numpy as np
import plotly.graph_objects as go


# --- 1. 物理计算核心 (简化模型用于可视化) ---
def generate_electron_paths(num_electrons, current_intensity, b_field_strength):
    """
    生成电子在洛伦兹力作用下的3D轨迹
    current_intensity: 模拟电流大小 (影响电子速度)
    b_field_strength: 磁场强度 (影响偏转曲率)
    """
    # 霍尔元件的尺寸 (长, 宽, 高)
    length, width, height = 10, 4, 1

    paths = []

    # 模拟电子初始位置 (从左侧射入)
    start_x = -length / 2

    for _ in range(num_electrons):
        # 随机分布在截面上
        start_y = np.random.uniform(-width / 2 + 0.5, width / 2 - 0.5)
        start_z = np.random.uniform(-height / 2 + 0.1, height / 2 - 0.1)

        # 时间步长
        t = np.linspace(0, 10, 100)

        # 1. X轴运动 (漂移速度，与电流成正比)
        # 设定一个基础速度 + 电流增益
        v_drift = 0.5 + (current_intensity * 0.5)
        x = start_x + v_drift * t

        # 2. Y轴偏转 (洛伦兹力 F = qvB -> 侧向加速度)
        # 简化物理模型：偏转量与 B * v 成正比
        # 假设磁场沿 Z 轴，电子受力沿 Y 轴
        deflection_factor = b_field_strength * v_drift * 0.1

        # 简单的二次曲线模拟偏转 y = y0 + 0.5 * a * t^2
        # 注意：加上边界检测，防止电子飞出导体太远（模拟碰壁堆积）
        deflection = 0.5 * deflection_factor * (t ** 2)
        y = start_y + deflection

        # 边界限制 (模拟霍尔电压产生时的电荷堆积，电子不能无限飞出)
        y = np.clip(y, -width / 2, width / 2)

        # 3. Z轴 (受限在薄片内，稍微有点随机抖动模拟热运动)
        z = np.full_like(t, start_z) + np.random.normal(0, 0.02, size=len(t))

        # 只保留还在导体长度内的点
        valid_idx = x <= length / 2
        paths.append((x[valid_idx], y[valid_idx], z[valid_idx]))

    return paths, (length, width, height)


# --- 2. Streamlit 页面布局 ---
st.set_page_config(page_title="3D 霍尔效应微观可视化", layout="wide")

st.title("🌌 3D 微观视界：霍尔效应可视化")
st.markdown("**AI 教学辅助演示：** 观察洛伦兹力如何改变载流子轨迹。")

col1, col2 = st.columns([1, 3])

with col1:
    st.header("🔬 实验参数控制")

    st.markdown("### 1. 励磁电流 (控制磁场 B)")
    mag_val = st.slider("磁场强度 (B)", -10.0, 10.0, 0.0, 0.5, format="%.1f T")
    if mag_val == 0:
        st.info("磁场为 0，电子沿直线运动。")
    elif mag_val > 0:
        st.warning(f"磁场向上 (+Z)，电子受力向一侧偏转。")
    else:
        st.warning(f"磁场向下 (-Z)，电子受力方向反转。")

    st.markdown("---")

    st.markdown("### 2. 工作电流 (控制漂移速度 v)")
    curr_val = st.slider("电流强度 (I)", 1.0, 5.0, 2.0, 0.5)

    st.markdown("---")
    st.caption("🔴 红色粒子代表电子 (带负电)")
    st.caption("🟦 蓝色框代表霍尔元件导体")

with col2:
    # --- 3. Plotly 3D 绘图 ---

    # 生成数据
    electron_paths, dims = generate_electron_paths(
        num_electrons=30,  # 粒子数量
        current_intensity=curr_val,
        b_field_strength=mag_val
    )
    L, W, H = dims

    fig = go.Figure()

    # 1. 绘制霍尔元件轮廓 (Wireframe Box)
    # 定义8个顶点的坐标来画框
    x_lines = [-L / 2, L / 2, L / 2, -L / 2, -L / 2, -L / 2, L / 2, L / 2, -L / 2, -L / 2, -L / 2, -L / 2, L / 2, L / 2,
               L / 2, L / 2]
    y_lines = [-W / 2, -W / 2, W / 2, W / 2, -W / 2, -W / 2, -W / 2, W / 2, W / 2, -W / 2, W / 2, W / 2, W / 2, -W / 2,
               -W / 2, W / 2]
    z_lines = [-H / 2, -H / 2, -H / 2, -H / 2, -H / 2, H / 2, H / 2, H / 2, H / 2, H / 2, H / 2, -H / 2, -H / 2, -H / 2,
               H / 2, H / 2]

    fig.add_trace(go.Scatter3d(
        x=x_lines, y=y_lines, z=z_lines,
        mode='lines',
        name='霍尔元件边界',
        line=dict(color='cyan', width=2, dash='dot'),
        opacity=0.3
    ))

    # 2. 绘制磁场向量 (大箭头)
    if abs(mag_val) > 0.1:
        arrow_z_start = -H
        arrow_z_end = H * 2 if mag_val > 0 else -H * 2
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[arrow_z_start, arrow_z_end],
            mode='lines+text',
            name='磁场 B',
            text=["", "B"],
            textposition="top center",
            line=dict(color='green', width=10)
        ))
        # 添加箭头头部的锥体 (用 Cone)
        fig.add_trace(go.Cone(
            x=[0], y=[0], z=[arrow_z_end],
            u=[0], v=[0], w=[1 if mag_val > 0 else -1],
            sizemode="absolute", sizeref=2, anchor="tail",
            showscale=False, colorscale=[[0, 'green'], [1, 'green']],
            name="B方向"
        ))

    # 3. 绘制电子轨迹
    for i, path in enumerate(electron_paths):
        px, py, pz = path
        fig.add_trace(go.Scatter3d(
            x=px, y=py, z=pz,
            mode='lines',
            line=dict(color='red', width=3),
            opacity=0.6,
            showlegend=False
        ))
        # 在末端加一个小球代表电子
        fig.add_trace(go.Scatter3d(
            x=[px[-1]], y=[py[-1]], z=[pz[-1]],
            mode='markers',
            marker=dict(size=4, color='red'),
            showlegend=False
        ))

    # 4. 布局设置
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='电流方向 (X)', range=[-L / 2 - 1, L / 2 + 1], showbackground=False),
            yaxis=dict(title='霍尔电压方向 (Y)', range=[-W, W], showbackground=False),
            zaxis=dict(title='磁场方向 (Z)', range=[-H * 3, H * 3], showbackground=False),
            aspectmode='manual',
            aspectratio=dict(x=2, y=1, z=0.5)  # 调整长宽比，让它看起来像个片状
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(x=0, y=1),
        paper_bgcolor="black",  # 黑色背景更有科技感
        plot_bgcolor="black",
        font=dict(color="white")
    )

    st.plotly_chart(fig, use_container_width=True)

    # 解释文本
    st.info("""
    **观察指南：**
    1. **无磁场时**：电子笔直通过导体。
    2. **增加磁场**：洛伦兹力 $F = q(\\vec{v} \\times \\vec{B})$ 产生作用。注意观察电子轨迹发生弯曲。
    3. **堆积效应**：电子最终打在侧壁（Y轴边界），这种电荷的不平衡分布就是**霍尔电压**的来源。
    """)