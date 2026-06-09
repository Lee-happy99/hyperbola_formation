# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# ------------------- 中文字体配置 -------------------
font_path = os.path.join(os.path.dirname(__file__), 'simhei.ttf')
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
else:
    st.warning("字体文件 simhei.ttf 未找到，将使用默认字体（可能显示方块）")
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.default'] = 'regular'

# ------------------- 页面配置 -------------------
st.set_page_config(page_title="双曲线几何定义与TDOA基础", layout="wide")
st.markdown("""
<style>
    .main .block-container { padding-top: 1rem !important; }
    h1 { font-size: 1.8rem !important; text-align: center !important; margin: 0rem 0rem 0.2rem 0rem; line-height: 1.2; }
    .stMarkdown p { text-align: center !important; margin-top: 0rem; margin-bottom: 0.5rem; line-height: 1.2; }
    div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] { gap: 0.2rem; }
    .stSlider { margin-top: -0.5rem; margin-bottom: -0.2rem; }
</style>
""", unsafe_allow_html=True)

st.title("📐 双曲线几何定义：到两定点距离差为常数")
st.markdown("**调整距离差常数 → 双曲线实时变化；拖动绘制进度 → 观察动点P满足 |PF₁ - PF₂| = 常数**")

# ------------------- 侧边栏参数 -------------------
st.sidebar.header("🔧 参数设置")
distance_diff = st.sidebar.slider("距离差常数 |PF₁ - PF₂| (km)", 0.5, 12.0, 6.0, 0.1)
progress = st.sidebar.slider("绘制进度 (t 参数)", 0.0, 1.0, 0.0, 0.01, help="0→1 逐步画出双曲线上的点")
st.sidebar.markdown("---")
col1, col2 = st.sidebar.columns(2)
with col1:
    x1 = st.slider("F₁ x", -10.0, 10.0, -5.0, 0.5)
with col2:
    y1 = st.slider("F₁ y", -10.0, 10.0, 0.0, 0.5)
col1, col2 = st.sidebar.columns(2)
with col1:
    x2 = st.slider("F₂ x", -10.0, 10.0, 5.0, 0.5)
with col2:
    y2 = st.slider("F₂ y", -10.0, 10.0, 0.0, 0.5)
show_axes = st.sidebar.checkbox("显示实轴和虚轴", value=True)

# ------------------- 几何计算 -------------------
F1 = np.array([x1, y1])
F2 = np.array([x2, y2])
center = (F1 + F2) / 2
c = np.linalg.norm(F2 - F1) / 2
a = distance_diff / 2
valid = a < c and a > 0

direction = (F2 - F1) / (2 * c) if c > 0 else np.array([1, 0])
rot = np.array([[direction[0], -direction[1]], [direction[1], direction[0]]])

def std_to_world(points_std):
    return (rot @ points_std.T).T + center

if valid:
    b = np.sqrt(c**2 - a**2)
    t_max = 3.0
    t_vals = np.linspace(-t_max, t_max, 400)
    # 右支
    x_right = a * np.cosh(t_vals)
    y_right = b * np.sinh(t_vals)
    right_branch = std_to_world(np.vstack((x_right, y_right)).T)
    # 左支
    x_left = -a * np.cosh(t_vals)
    y_left = b * np.sinh(t_vals)
    left_branch = std_to_world(np.vstack((x_left, y_left)).T)
    
    n = len(t_vals)
    idx = int(n * progress)
    right_part = right_branch[:idx]
    left_part = left_branch[:idx]
    
    if progress > 0:
        t_curr = t_vals[idx-1] if idx>0 else t_vals[0]
        p_std = np.array([a * np.cosh(t_curr), b * np.sinh(t_curr)])
        P = std_to_world(p_std.reshape(1,2))[0]
        d1 = np.linalg.norm(P - F1)
        d2 = np.linalg.norm(P - F2)
        actual_diff = abs(d1 - d2)
    else:
        P = None
        actual_diff = None
else:
    b = None
    right_branch = left_branch = right_part = left_part = None
    P = None

# ------------------- 动态坐标轴范围 -------------------
all_x = [x1, x2]
all_y = [y1, y2]
if valid:
    if len(right_branch) > 0:
        all_x.extend(right_branch[:,0])
        all_y.extend(right_branch[:,1])
    if len(left_branch) > 0:
        all_x.extend(left_branch[:,0])
        all_y.extend(left_branch[:,1])
if P is not None:
    all_x.append(P[0])
    all_y.append(P[1])
if all_x:
    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)
    margin_x = max(1.0, (x_max - x_min) * 0.15)
    margin_y = max(1.0, (y_max - y_min) * 0.15)
    xlim = (x_min - margin_x, x_max + margin_x)
    ylim = (y_min - margin_y, y_max + margin_y)
else:
    xlim, ylim = (-10, 10), (-8, 8)

# ------------------- 绘图 -------------------
fig, ax = plt.subplots(figsize=(8, 7))
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# 焦点（蓝色圆点，大小减半）
ax.plot(F1[0], F1[1], 'bo', markersize=5)
ax.plot(F2[0], F2[1], 'bo', markersize=5)
# 焦点标注：大幅增加偏移量，确保远离圆点
ax.text(F1[0] - 1.2, F1[1] - 2.5, r"$F_1$", color='blue', fontsize=10, ha='center', weight='bold')
ax.text(F2[0] + 1.2, F2[1] - 2.5, r"$F_2$", color='blue', fontsize=10, ha='center', weight='bold')

if valid:
    # 已绘制部分：右支实线，左支虚线
    if len(right_part) > 0:
        ax.plot(right_part[:,0], right_part[:,1], 'r-', linewidth=2, label='双曲线 (右支, 实线)')
    if len(left_part) > 0:
        ax.plot(left_part[:,0], left_part[:,1], 'r--', linewidth=2, label='双曲线 (左支, 虚线)')
    
    # 未绘制部分（灰色虚线）
    if progress < 1.0:
        if len(right_branch) > 0:
            ax.plot(right_branch[:,0], right_branch[:,1], 'gray', linewidth=1, alpha=0.3, linestyle='--')
        if len(left_branch) > 0:
            ax.plot(left_branch[:,0], left_branch[:,1], 'gray', linewidth=1, alpha=0.3, linestyle='--')
    
    # 实轴和虚轴
    if show_axes:
        v_right_std = np.array([a, 0])
        v_left_std = np.array([-a, 0])
        v_right = std_to_world(v_right_std.reshape(1,2))[0]
        v_left = std_to_world(v_left_std.reshape(1,2))[0]
        ax.plot([v_left[0], v_right[0]], [v_left[1], v_right[1]], 'k-', linewidth=1.5, alpha=0.7, label='实轴')
        perp = np.array([-direction[1], direction[0]])
        end1 = center + perp * b
        end2 = center - perp * b
        ax.plot([end1[0], end2[0]], [end1[1], end2[1]], 'k--', linewidth=1.2, alpha=0.7, label='虚轴')
        ax.plot(v_right[0], v_right[1], 'ko', markersize=4)
        ax.plot(v_left[0], v_left[1], 'ko', markersize=4)
        ax.plot(center[0], center[1], 'k+', markersize=8, mew=1.5)
    
    # 动点 P（绿色，稍大一点以突出）
    if P is not None:
        ax.plot(P[0], P[1], 'go', markersize=6, label='动点 P')
        ax.text(P[0]+0.3, P[1]+0.3, "P", color='green', fontsize=10, weight='bold')
        ax.plot([P[0], F1[0]], [P[1], F1[1]], 'g--', linewidth=1.5, alpha=0.8)
        ax.plot([P[0], F2[0]], [P[1], F2[1]], 'g--', linewidth=1.5, alpha=0.8)
        # 距离差文本框移至 P 点右上方更远处，避免遮挡
        ax.text(P[0] + 1.8, P[1] + 1.5, r"$|PF_1-PF_2| = {:.2f}$ km".format(actual_diff),
                color='green', fontsize=9, ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # 参数信息框（左上角，仅显示距离差常数和半实轴 a）
    info_text = f"距离差常数 = {distance_diff:.2f} km\n半实轴 a = {a:.2f}"
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
else:
    ax.text(0.1, 0.5, "参数不合理：距离差 ≥ 两焦点距离，无法形成双曲线", transform=ax.transAxes, color='red')

ax.legend(loc='upper right', fontsize=8)
st.pyplot(fig, use_container_width=True)

with st.expander("📖 双曲线几何定义与 TDOA 原理衔接（点击展开）"):
    st.markdown(r"""
    - **双曲线定义**：平面内到两个定点（焦点）的距离之差的绝对值为常数（$2a$，且 $2a < |F_1F_2|$）的点的轨迹。  
    - **实轴**：连接两顶点的线段，长度 $2a$。  
    - **虚轴**：通过中心垂直于实轴的线段，长度 $2b$，满足 $c^2 = a^2 + b^2$。  
    - **线型说明**：右支用**红色实线**，左支用**红色虚线**，便于区分两支。  
    - **TDOA 定位基础**：两个侦察站 $F_1, F_2$ 测得到目标的距离差 $\Delta d = c \cdot \Delta t$ → 目标位于以 $F_1, F_2$ 为焦点的双曲线上。引入第三个站，两条双曲线相交即得目标位置。  
    - **操作提示**：① 调整距离差常数，观察双曲线形状变化；② 拖动“绘制进度”，观察动点 $P$ 移动时虚线长度差始终等于常数；③ 可勾选/取消显示实轴虚轴。
    """)
