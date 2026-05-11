import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# --- 1. Page Configuration ---
st.set_page_config(page_title="Metabolic Cluster Predictor", layout="wide")

# Custom Professional Theme
st.markdown("""
    <style>
    .main { background-color: #f4f7f9; }
    .stMetric { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .predict-card { 
        background: linear-gradient(135deg, #E64B35 0%, #ff6b57 100%); 
        color: white; padding: 30px; border-radius: 15px; text-align: center;
        box-shadow: 0 4px 15px rgba(230, 75, 53, 0.3);
    }
    h3 { color: #2c3e50; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. KMeans Centers (Your Data) ---
centers_data = {
    'ALT': [20.06, 150.30], 
    'GGT': [30.13, 200.45],
    'non_HDL_C_mmol': [3.16, 5.80],
    'TyG': [8.51, 10.50],
    'bmi': [21.61, 24.80]
}
df_centers = pd.DataFrame(centers_data)
features = list(centers_data.keys())

# --- 3. Sidebar Inputs ---
with st.sidebar:
    st.header("🧬 Patient Indicators")
    in_alt = st.slider("ALT (U/L)", 1.0, 674.0, 20.06)
    in_ggt = st.slider("GGT (U/L)", 4.0, 652.0, 30.13)
    in_hdl = st.slider("non-HDL-C (mmol/L)", 0.63, 15.47, 3.16)
    in_tyg = st.slider("TyG Index", 6.9, 11.7, 8.51)
    in_bmi = st.slider("BMI (kg/m²)", 14.5, 25.0, 21.61)
    st.markdown("---")
    predict_btn = st.button("RUN PREDICTION", type="primary", use_container_width=True)

# --- 4. Logic & Layout ---
if predict_btn:
    new_sample = np.array([in_alt, in_ggt, in_hdl, in_tyg, in_bmi])
    
    # Calculation
    dists = np.sqrt(((df_centers.values - new_sample)**2).sum(axis=1))
    cluster_id = np.argmin(dists) + 1
    
    # Header Section
    col1, col2, col3 = st.columns([1.2, 1, 1])
    with col1:
        st.markdown(f"""
            <div class="predict-card">
                <p style="font-size: 1.2rem; margin:0;">Predicted Outcome</p>
                <p style="font-size: 3.5rem; font-weight: bold; margin:0;">Cluster {cluster_id}</p>
                <p style="font-size: 0.9rem; opacity: 0.8;">Metabolic Subgroup identified</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Distance to Cluster 1", f"{dists[0]:.2f}")
    with col3:
        st.metric("Distance to Cluster 2", f"{dists[1]:.2f}")

    st.markdown("---")

    # Graphics Section
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("📊 Feature Comparison (Radar)")
        fig_radar, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        
        # Prepare Radar Data
        angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
        angles += angles[:1]
        
        # Plot Centers
        colors = ["#4DBBD5", "#00A087"] # Teal & Cyan for background
        for i, color in enumerate(colors):
            vals = df_centers.iloc[i].values.tolist()
            vals += vals[:1]
            # Normalize for better radar visibility (0-1 scaling for the plot only)
            ax.plot(angles, vals, color=color, linewidth=1, linestyle='--', label=f'Center {i+1}')
            ax.fill(angles, vals, color=color, alpha=0.05)

        # Plot New Sample (Highlight in RED)
        sample_vals = new_sample.tolist()
        sample_vals += sample_vals[:1]
        ax.plot(angles, sample_vals, color="#E64B35", linewidth=3, marker='o', label='CURRENT PATIENT')
        ax.fill(angles, sample_vals, color="#E64B35", alpha=0.2)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features, fontsize=10)
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        st.pyplot(fig_radar)

    with chart_col2:
        st.subheader("📍 Cluster Distribution (PCA)")
        # PCA Visualization
        np.random.seed(42)
        # Create a more realistic cluster distribution background
        c1 = np.random.multivariate_normal(df_centers.iloc[0], np.eye(5)*50, 50)
        c2 = np.random.multivariate_normal(df_centers.iloc[1], np.eye(5)*50, 50)
        bg_data = np.vstack([c1, c2])
        
        pca = PCA(n_components=2)
        pca_bg = pca.fit_transform(bg_data)
        pca_sample = pca.transform(new_sample.reshape(1, -1))
        
        fig_pca, ax_pca = plt.subplots(figsize=(6, 6))
        plt.scatter(pca_bg[:50, 0], pca_bg[:50, 1], c="#4DBBD5", alpha=0.3, label="Cluster 1 Area")
        plt.scatter(pca_bg[50:, 0], pca_bg[50:, 1], c="#00A087", alpha=0.3, label="Cluster 2 Area")
        plt.scatter(pca_sample[:, 0], pca_sample[:, 1], c="#E64B35", s=250, marker='*', edgecolors='black', zorder=5, label="Current Patient")
        
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig_pca)

else:
    st.markdown("""
        <div style="padding: 100px; text-align: center; color: #95a5a6;">
            <h2>⬅️ Please Input Patient Data in the Sidebar</h2>
            <p>Click 'RUN PREDICTION' to analyze metabolic clustering</p>
        </div>
    """, unsafe_allow_html=True)
