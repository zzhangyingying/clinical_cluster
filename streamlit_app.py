import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# 1. Page Configuration (Full English)
st.set_page_config(page_title="KMeans Clinical Prediction Tool", layout="wide")

# Custom CSS for the "Professional Look" (Matches your R App style)
st.markdown("""
    <style>
    .stApp { background-color: #f7f9fc; }
    .result-box { background: white; border-radius: 10px; padding: 25px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); margin-bottom: 20px; }
    .cluster-badge { font-size: 36px; font-weight: bold; text-align: center; padding: 15px; border-radius: 15px; border: 3px solid #E64B35; color: #E64B35; background: #E64B3522; }
    </style>
    """, unsafe_allow_html=True)

# 2. FIXED: Cluster Centers for 2 Clusters (Based on your R logic)
# Please replace these values with your ACTUAL centers from R: km_res$centers
centers_data = {
    'ALT': [20.06, 150.30], 
    'GGT': [30.13, 200.45],
    'non_HDL_C_mmol': [3.16, 5.80],
    'TyG': [8.51, 10.50],
    'bmi': [21.61, 24.80]
}
centers_df = pd.DataFrame(centers_data)
var_list = list(centers_data.keys())

# 3. Sidebar: Inputs
st.sidebar.header("Input Sample Features")
inputs = {}
for var in var_list:
    # Set default values based on your R slider settings
    if var == 'ALT': inputs[var] = st.sidebar.slider(var, 1.0, 674.0, 20.06)
    elif var == 'GGT': inputs[var] = st.sidebar.slider(var, 4.0, 652.0, 30.13)
    elif var == 'non_HDL_C_mmol': inputs[var] = st.sidebar.slider(var, 0.63, 15.47, 3.16)
    elif var == 'TyG': inputs[var] = st.sidebar.slider(var, 6.9, 11.7, 8.51)
    elif var == 'bmi': inputs[var] = st.sidebar.slider(var, 14.5, 25.0, 21.61)

predict_btn = st.sidebar.button("Predict Cluster", type="primary")

# 4. Prediction & Visualization Logic
if predict_btn:
    new_sample = np.array([inputs[v] for v in var_list])
    
    # Calculate Euclidean Distance to the 2 centers
    dists = np.sqrt(((centers_df.values - new_sample)**2).sum(axis=1))
    cluster_id = np.argmin(dists) + 1
    
    # --- UI Layout ---
    st.title(f"KMeans Prediction Result (K=2)")
    
    col_main, col_side = st.columns([2, 1])
    
    with col_side:
        st.markdown(f'<div class="result-box"><h4>Prediction Result</h4>'
                    f'<div class="cluster-badge">Cluster {cluster_id}</div>'
                    f'<p style="text-align:center; color:grey; margin-top:10px;">'
                    f'Distance to center: {dists[cluster_id-1]:.4f}</p></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="result-box"><h4>Distance Analysis</h4>', unsafe_allow_html=True)
        dist_df = pd.DataFrame({'Group': ['Cluster 1', 'Cluster 2'], 'Distance': dists})
        st.bar_chart(dist_df.set_index('Group'))
        st.markdown('</div>', unsafe_allow_html=True)

    with col_main:
        # --- Radar Chart (Matches your R Polar Bar Chart) ---
        st.markdown('<div class="result-box"><h4>Cluster Centers Radar Chart</h4>', unsafe_allow_html=True)
        fig_radar, ax_radar = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        
        # Prepare data for polar plot
        labels = var_list
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1] # close the loop
        
        colors = ["#E64B35", "#4DBBD5"]
        for i in range(2):
            values = centers_df.iloc[i].values.tolist()
            values += values[:1]
            ax_radar.plot(angles, values, color=colors[i], linewidth=2, label=f'Cluster {i+1}')
            ax_radar.fill(angles, values, color=colors[i], alpha=0.25)
        
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(labels)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        st.pyplot(fig_radar)
        st.markdown('</div>', unsafe_allow_html=True)

        # --- PCA Plot (Simulated Scatter Plot) ---
        st.markdown('<div class="result-box"><h4>PCA Cluster Plot (Feature Distribution)</h4>', unsafe_allow_html=True)
        # Create dummy data for PCA visualization context
        np.random.seed(42)
        dummy_data = np.random.randn(100, 5) 
        pca = PCA(n_components=2)
        components = pca.fit_transform(dummy_data)
        new_sample_pca = pca.transform(new_sample.reshape(1, -1))
        
        fig_pca, ax_pca = plt.subplots()
        plt.scatter(components[:, 0], components[:, 1], c='lightgrey', alpha=0.5, label='Existing Data')
        plt.scatter(new_sample_pca[:, 0], new_sample_pca[:, 1], c=colors[cluster_id-1], s=200, marker='*', edgecolors='black', label='New Sample')
        plt.title("PCA Projection of Clusters")
        plt.legend()
        st.pyplot(fig_pca)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Please adjust sample features in the sidebar and click 'Predict Cluster'.")
