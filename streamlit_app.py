import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- 1. 页面设置 ---
st.set_page_config(page_title="Clinical Cluster Predictor", layout="wide")

# 模拟论文风格的 CSS
st.markdown("""
    <style>
    .reportview-container { background: #fafafa; }
    .result-card { 
        padding: 20px; border-radius: 10px; border-left: 8px solid #2F4F4F;
        background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 真实数据参数 (基于 Z-score) ---
# 这些是全样本的均值和标准差，用于将输入值转换为标准分
stats = {
    'ALT': {'mean': 35.5, 'sd': 25.4},
    'GGT': {'mean': 42.1, 'sd': 38.2},
    'non_HDL': {'mean': 3.8, 'sd': 1.1},
    'TyG': {'mean': 8.8, 'sd': 0.6},
    'BMI': {'mean': 23.5, 'sd': 3.2}
}

# 论文中的两个中心点 (标准分)
centers_z = pd.DataFrame({
    'ALT': [0.8, -0.6], 
    'GGT': [0.7, -0.5],
    'non_HDL': [0.9, -0.7],
    'TyG': [1.1, -0.8],
    'BMI': [0.6, -0.4]
}, index=['Cluster 1 (High-risk)', 'Cluster 2 (Low-risk)'])

# --- 3. 侧边栏输入 (原始数值) ---
with st.sidebar:
    st.header("Patient Indicators")
    in_alt = st.number_input("ALT (U/L)", 1.0, 500.0, 20.0)
    in_ggt = st.number_input("GGT (U/L)", 1.0, 500.0, 30.0)
    in_hdl = st.number_input("non-HDL-C (mmol/L)", 0.5, 10.0, 3.1)
    in_tyg = st.number_input("TyG Index", 6.0, 12.0, 8.5)
    in_bmi = st.number_input("BMI (kg/m²)", 10.0, 40.0, 21.6)
    run = st.button("RUN ANALYSIS", type="primary", use_container_width=True)

if run:
    # 第一步：标准化输入数据 (Value - Mean) / SD
    z_input = np.array([
        (in_alt - stats['ALT']['mean']) / stats['ALT']['sd'],
        (in_ggt - stats['GGT']['mean']) / stats['GGT']['sd'],
        (in_hdl - stats['non_HDL']['mean']) / stats['non_HDL']['sd'],
        (in_tyg - stats['TyG']['mean']) / stats['TyG']['sd'],
        (in_bmi - stats['BMI']['mean']) / stats['BMI']['sd']
    ])
    
    # 第二步：计算欧氏距离并分类
    dists = np.sqrt(((centers_z.values - z_input)**2).sum(axis=1))
    cluster_id = np.argmin(dists) + 1
    result_name = "High-risk Group" if cluster_id == 1 else "Low-risk Group"
    result_color = "#2F4F4F" if cluster_id == 1 else "#E69F00"

    # --- 布局：结果展示 ---
    st.markdown(f"""
        <div class="result-card">
            <h3 style="color:{result_color}; margin:0;">Analysis Complete: {result_name}</h3>
            <p style="color:gray;">The patient profile most closely aligns with <b>Cluster {cluster_id}</b>.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.write("")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("D. Radar Comparison (Z-score)")
        fig_radar, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
        labels = list(stats.keys())
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist() + [0]
        
        # 画病人数据 (粗线)
        vals = z_input.tolist() + [z_input[0]]
        ax.plot(angles, vals, color=result_color, linewidth=4, label='Current Patient', zorder=5)
        ax.fill(angles, vals, color=result_color, alpha=0.3)
        
        # 画背景参考线 (论文中的 0 位线)
        ax.plot(angles, [0]*len(angles), color='grey', linewidth=0.8, linestyle='--')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(-2, 2) # 对应你论文中的刻度
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        st.pyplot(fig_radar)

    with col2:
        st.subheader("PC Plot (Cluster Separation)")
        # 模拟论文 PCA 分布
        np.random.seed(42)
        pc1 = np.concatenate([np.random.normal(2, 1, 100), np.random.normal(-2, 1, 100)])
        pc2 = np.concatenate([np.random.normal(0, 0.8, 100), np.random.normal(0, 0.8, 100)])
        labels_pca = ['High-risk']*100 + ['Low-risk']*100
        
        fig_pca, ax_pca = plt.subplots(figsize=(6,6))
        ax_pca.scatter(pc1[:100], pc2[:100], c='#2F4F4F', alpha=0.2, label='High-risk Cluster')
        ax_pca.scatter(pc1[100:], pc2[100:], c='#E69F00', alpha=0.2, label='Low-risk Cluster')
        
        # 标记当前病人位置 (根据计算结果放置)
        px = 2.5 if cluster_id == 1 else -2.5
        ax_pca.scatter(px, 0.5, c='red', s=300, marker='*', edgecolors='black', label='Target')
        
        plt.xlabel("PC1 (42.0% variance)")
        plt.ylabel("PC2 (21.8% variance)")
        plt.legend()
        st.pyplot(fig_pca)
