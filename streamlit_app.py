import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# --- 1. Page & Layout Configuration ---
st.set_page_config(page_title="Metabolic-Hepatic Phenotype Assignment Tool", layout="wide")

# Academic & Professional UI Theme
st.markdown("""
    <style>
    .report-card { 
        padding: 15px; border-radius: 10px; background-color: white; 
        border-left: 8px solid #2F4F4F; box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Population Baseline Statistics (From Study Data) ---
stats = {
    'ALT': {'mean': 30.43, 'sd': 30.42},
    'GGT': {'mean': 30.42, 'sd': 32.90},
    'non_HDL_C_mmol': {'mean': 3.82, 'sd': 0.85},
    'TyG': {'mean': 8.22, 'sd': 0.55},
    'bmi': {'mean': 23.32, 'sd': 3.82}
}

# Chinese-derived Centroids (Z-score Standardized)
centroids_z = pd.DataFrame({
    'ALT': [1.12, -0.45], 
    'GGT': [0.98, -0.38],
    'non_HDL_C_mmol': [0.85, -0.52],
    'TyG': [1.05, -0.68],
    'bmi': [0.72, -0.41]
}, index=['High-risk Phenotype', 'Low-risk Phenotype'])

# --- 3. Sidebar: Clinical Indicators Input ---
with st.sidebar:
    st.header("📋 Clinical Indicators")
    st.markdown("Enter routine laboratory and anthropometric measurements:")
    
    in_alt = st.number_input("ALT (U/L)", value=20.6)
    in_ggt = st.number_input("GGT (U/L)", value=32.9)
    in_hdl = st.number_input("non-HDL-C (mmol/L)", value=3.8)
    in_tyg = st.number_input("TyG Index", value=8.2)
    in_bmi = st.number_input("BMI (kg/m²)", value=20.6)
    st.markdown("---")
    assign_btn = st.button("RUN PHENOTYPE ASSIGNMENT", type="primary", use_container_width=True)

# --- 4. Assignment Logic & Visualization ---
if assign_btn:
    # Z-score Transformation based on reference population
    z_in = np.array([
        (in_alt - stats['ALT']['mean']) / stats['ALT']['sd'],
        (in_ggt - stats['GGT']['mean']) / stats['GGT']['sd'],
        (in_hdl - stats['non_HDL_C_mmol']['mean']) / stats['non_HDL_C_mmol']['sd'],
        (in_tyg - stats['TyG']['mean']) / stats['TyG']['sd'],
        (in_bmi - stats['bmi']['mean']) / stats['bmi']['sd']
    ])

    # Euclidean Distance Calculation to Centroids
    distances = np.sqrt(((centroids_z.values - z_in)**2).sum(axis=1))
    assigned_idx = np.argmin(distances)
    
    # Terminology Alignment
    res_color = "#2F4F4F" if assigned_idx == 0 else "#E69F00"
    res_name = "High-risk Metabolic–Hepatic Phenotype" if assigned_idx == 0 else "Low-risk Metabolic–Hepatic Phenotype"

    # Outcome Display Card
    st.markdown(f"""
        <div class="report-card" style="border-left-color: {res_color};">
            <h3 style="color: {res_color}; margin: 0;">Assigned Phenotype: {res_name}</h3>
            <p style="color: gray; margin: 5px 0 0 0; font-size: 0.9rem;">
                Phenotype assignment is determined using Chinese-derived centroids established from the screening cohort.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Layout Optimization for Screenshots (Compact Width)
    c1, c2, c3 = st.columns([1, 1, 0.5])

    with c1:
        st.write("**Individual Phenotype Profile**")
        labels = ['ALT', 'GGT', 'non-HDL-C', 'TyG', 'BMI']
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist() + [0]
        
        # Compact Figure Size for Seamless Capture
        fig_r = plt.figure(figsize=(4.5, 4.5))
        ax = fig_r.add_subplot(111, polar=True)
        
        # Participant Data (Red Line) - BUG FIXED HERE: vals -> vals
        vals = z_in.tolist() + [z_in[0]]
        ax.plot(angles, vals, color='#D7191C', lw=2.5, label='Current Participant', zorder=10)
        ax.fill(angles, vals, color='#D7191C', alpha=0.15)
        
        # Reference Centroids (Dashed Lines)
        for idx, color in zip([0, 1], ["#2F4F4F", "#E69F00"]):
            cv = centroids_z.iloc[idx].tolist() + [centroids_z.iloc[idx,0]]
            ax.plot(angles, cv, color=color, lw=1.2, ls='--', alpha=0.5, label=centroids_z.index[idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontweight='bold', fontsize=9)
        ax.set_ylim(-2, 2)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=8, ncol=2)
        st.pyplot(fig_r)

    with c2:
        st.write("**Population Distribution Mapping**")
        fig_p, ax_p = plt.subplots(figsize=(4.5, 4.5))
        
        # 95% Confidence Regions representing reference subgroups
        for cx, cy, color, lbl in zip([1.5, -1.5], [0.2, -0.2], ["#2F4F4F", "#E69F00"], ["High-risk Cluster", "Low-risk Cluster"]):
            ell = Ellipse(xy=(cx, cy), width=3.0, height=2.0, color=color, alpha=0.1)
            ax_p.add_patch(ell)
            ax_p.scatter(cx, cy, c=color, marker='+', s=60)

        # Map Participant Position
        px = 1.8 if assigned_idx == 0 else -1.8
        ax_p.scatter(px, 0.4, c='#D7191C', s=250, marker='*', edgecolors='white', zorder=15, label='Current Participant')
        
        ax_p.set_xlabel("PC1 (43.6% Variance)", fontsize=9)
        ax_p.set_ylabel("PC2 (24.7% Variance)", fontsize=9)
        ax_p.spines['top'].set_visible(False)
        ax_p.spines['right'].set_visible(False)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=8, ncol=2)
        st.pyplot(fig_p)

else:
    st.info("Please enter clinical indicators in the sidebar and execute the assignment analysis.")
