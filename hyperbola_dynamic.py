# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 08:57:01 2026

@author: ASUS
"""

# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# 字体配置
font_path = os.path.join(os.path.dirname(__file__), 'simhei.ttf')
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
else:
    st.warning("字体文件 simhei.ttf 未找到")
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="双曲线动态演示", layout="wide")
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem !important; }
    h1 { font-size: 1.8rem !important; text-align: center !important; margin: 0 0 0.2rem 0; line-height: 1.2; }
    .stMarkdown p { text-align: center !important; margin: 0 0 0.5rem 0; line-height: 1.2; }
</style>
""", unsafe_allow_html=True)

st.title("📐 双曲线定义：到两定点距离差为常数")
st.markdown("**调整下方滑块，观察距离差常数变化对双曲线形状的影响**")

# 侧边栏参数
st.sidebar.header("🔧 参数设置")
x1 = st.sidebar.slider("焦点 F₁ x (km)", -10.0, 10.0, -5.0, 0.2)
y1 = st.sidebar.slider("焦点 F₁ y (km)", -8.0, 8.0, 0.0, 0.2)
x2 = st.sidebar.slider("焦点 F₂ x (km)", -10.0, 10.0, 5.0, 0.2)
y2 = st.sidebar.slider("焦点 F₂ y (km)", -8.0, 8.0, 0.0, 0.2)

F1 = np.array([x1, y1])
F2 = np.array([x2, y2])
c = np.linalg.norm(F2 - F1) / 2  # 半焦距

# 距离差常数（2a），必须小于 2c
max_diff = 2 * c - 0.1
if max_diff <= 0:
    max_diff = 0.1
distance_diff = st.sidebar.slider("距离差常数 |PF₁ - PF₂| (km)", 0.1, float(max_diff), min(2.0, max_diff), 0.1,
                                  help="必须小于两焦点距离")

a = distance_diff / 2
valid = a < c and a > 0

# 计算双曲线点集（标准坐标系参数化）
def get_hyperbola_points(F1, F2, a, c, num=300):
    if a >= c or a <= 0:
        return [], []
    b = np.sqrt(c**2 - a**2)
    # 参数 t 范围 -3 到 3 足以覆盖显示区域
    t = np.linspace(-3, 3, num)
    x_std_right = a * np.cosh(t)
    y_std_right = b * np.sinh(t)
    x_std_left = -a * np.cosh(t)
    y_std_left = b * np.sinh(t)
    
    # 坐标变换：将标准坐标系（中心为两焦点中点，x轴沿F1F2）转换到世界坐标系
    mid = (F1 + F2) / 2
    direction = (F2 - F1) / (2 * c)  # 单位向量
    # 旋转矩阵（x轴旋转到direction）
    rot = np.array([[direction[0], -direction[1]], [direction[1], direction[0]]])
    points_right = np.vstack((x_std_right, y_std_right)).T
    points_left = np.vstack((x_std_left, y_std_left)).T
    world_right = (rot @ points_right.T).T + mid
    world_left = (rot @ points_left.T).T + mid
    return world_right, world_left

if valid:
    right, left = get_hyperbola_points(F1, F2, a, c)
else:
    right, left = [], []

# 绘图
fig, ax = plt.subplots(figsize=(8, 7))
ax.set_xlim(-10, 10)
ax.set_ylim(-8, 8)
ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# 绘制焦点
ax.plot(F1[0], F1[1], 'bo', markersize=10, label='焦点 F₁')
ax.plot(F2[0], F2[1], 'bo', markersize=10, label='焦点 F₂')
ax.text(F1[0], F1[1]-1.2, "F₁", color='blue', ha='center', fontsize=10)
ax.text(F2[0], F2[1]-1.2, "F₂", color='blue', ha='center', fontsize=10)

# 绘制双曲线
if valid and len(right) > 0 and len(left) > 0:
    ax.plot(right[:,0], right[:,1], 'r-', linewidth=2, label='双曲线')
    ax.plot(left[:,0], left[:,1], 'r-', linewidth=2)
    # 显示参数信息
    info = f"两焦点距离 = {2*c:.2f} km\n距离差 = {distance_diff:.2f} km\n半实轴 a = {a:.2f} km"
    ax.text(0.05, 0.95, info, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
else:
    ax.text(0.1, 0.5, "距离差必须小于两焦点距离，无法形成双曲线", transform=ax.transAxes, color='red')

ax.legend(loc='upper right')
st.pyplot(fig, use_container_width=True)

with st.expander("📖 教学说明"):
    st.markdown(r"""
    - **双曲线定义**：$|PF_1 - PF_2| = 2a$（常数），且 $2a < |F_1F_2|$。  
    - **动态演示**：拖动滑块改变焦点位置或距离差常数，双曲线形状实时变化。  
    - **关键观察**：常数越大（越接近两焦点距离），双曲线开口越扁；常数越小，开口越宽，越接近两支的渐近线。  
    - **与TDOA的联系**：TDOA定位中，两个侦察站即为焦点，时差转换的距离差即为常数，目标位于双曲线上。
    """)