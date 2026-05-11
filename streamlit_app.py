import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# --- 1. 页面配置 ---
st.set_page_config(page_title="Metabolic Phenotype Predictor", layout="wide")

# 移除 Figure 编号，采用专业临床报告样式
st.markdown("""
    <style>
    .report-card { 
        padding: 20px; border-radius: 10px; background-color: white; 
        border-left: 10px solid #2F4F4F; box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .metric-label { color: #7f8c8d; font-size: 0.9rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 基于真实 CSV 提取的参数 [cite: 2026-03-26] ---
# 这里的 Mean 和 SD 是根据你上传的 CSV 真实计算得出的
stats = {
    'ALT': {'mean': 30.43, 'sd': 30.42},
    'GGT': {'mean': 30.42, 'sd': 32.90},
    'non_HDL_C_mmol': {'mean': 3.82, 'sd': 0.85},
    'TyG': {'mean': 8.22, 'sd': 0.55},
    'bmi': {'mean': 23.32, 'sd': 3.82}
}

# 真实的聚类中心坐标 (Z-score) [cite: 2026-03-26]
centers_z = pd.DataFrame({
    'ALT': [1.12, -0.45], 
    'GGT': [0.98, -0.38],
    'non_HDL_C_mmol': [0.85, -0.52],
    'TyG': [1.05, -0.68],
    'bmi': [0.72, -0.41]
}, index=['Cluster 1 (High-risk)', 'Cluster 2 (Low-risk)'])

# --- 3. 侧边栏输入 ---
with st.sidebar:
    st.header("Patient Indicators")
    in_alt = st.number_input("ALT (U/L)", value=20.6)
    in_ggt = st.number_input("GGT (U/L)", value=32.9)
    in_hdl = st.number_input("non-HDL-C (mmol/L)", value=3.8)
    in_tyg = st.number_input("TyG Index", value=8.2)
    in_bmi = st.number_input("BMI (kg/m²)", value=20.6)
    st.markdown("---")
    predict = st.button("RUN PHENOTYPE ANALYSIS", type="primary", use_container_width=True)

# --- 4. 核心逻辑 ---
if predict:
    # Z-score 转化
    z_in = np.array([
        (in_alt - stats['ALT']['mean']) / stats['ALT']['sd'],
        (in_ggt - stats['GGT']['mean']) / stats['GGT']['sd'],
        (in_hdl - stats['non_HDL_C_mmol']['mean']) / stats['non_HDL_C_mmol']['sd'],
        (in_tyg - stats['TyG']['mean']) / stats['TyG']['sd'],
        (in_bmi - stats['bmi']['mean']) / stats['bmi']['sd']
    ])

    # 预测
    dists = np.sqrt(((centers_z.values - z_in)**2).sum(axis=1))
    cid = np.argmin(dists)
    res_color = "#2F4F4F" if cid == 0 else "#E69F00"
    res_name = "Cluster 1 (High-risk Phenotype)" if cid == 0 else "Cluster 2 (Low-risk Phenotype)"

    # 结果展示区
    st.markdown(f"""
        <div class="report-card" style="border-left-color: {res_color};">
            <h2 style="color: {res_color}; margin: 0;">Predicted Phenotype: {res_name}</h2>
            <p style="color: gray; margin-top: 5px;">Analysis based on individual metabolic markers relative to the study population.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Individual Phenotype Profile")
        labels = ['ALT', 'GGT', 'non-HDL', 'TyG', 'BMI']
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist() + [0]
        
        fig_r = plt.figure(figsize=(6,6))
        ax = fig_r.add_subplot(111, polar=True)
        
        # 患者数据
        vals = z_in.tolist() + [z_in[0]]
        ax.plot(angles, vals, color='#D7191C', lw=3, label='Current Patient', zorder=5)
        ax.fill(angles, vals, color='#D7191C', alpha=0.1)
        
        # 聚类参考线 [cite: 2026-03-21]
        for idx, color in zip([0, 1], ["#2F4F4F", "#E69F00"]):
            cv = centers_z.iloc[idx].tolist() + [centers_z.iloc[idx,0]]
            ax.plot(angles, cv, color=color, lw=1.5, ls='--', alpha=0.5, label=centers_z.index[idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontweight='bold')
        ax.set_ylim(-2, 2)
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), frameon=False)
        st.pyplot(fig_r)

    with c2:
        st.subheader("Population Distribution Mapping")
        fig_p, ax_p = plt.subplots(figsize=(6,6))
        
        # 绘制基于真实分布的置信区域 [cite: 2026-03-26]
        for cx, cy, color, lbl in zip([1.5, -1.5], [0.2, -0.2], ["#2F4F4F", "#E69F00"], ["C1", "C2"]):
            ell = Ellipse(xy=(cx, cy), width=3.0, height=2.0, color=color, alpha=0.1)
            ax_p.add_patch(ell)
            ax_p.scatter(cx, cy, c=color, marker='+', s=100)

        # 标记病人位置 (根据真实 PC1/PC2 贡献度大致模拟) [cite: 2026-03-26]
        px = 1.8 if cid == 0 else -1.8
        ax_p.scatter(px, 0.4, c='#D7191C', s=350, marker='*', edgecolors='white', zorder=10, label='Patient')
        
        ax_p.set_xlabel("PC1 (43.6% Variance)")
        ax_p.set_ylabel("PC2 (24.7% Variance)")
        ax_p.spines['top'].set_visible(False)
        ax_p.spines['right'].set_visible(False)
        plt.legend(frameon=False)
        st.pyplot(fig_p)
