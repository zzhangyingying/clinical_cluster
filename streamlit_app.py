import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Page Configuration
st.set_page_config(page_title="KMeans Clinical Prediction Tool", layout="wide")

# 2. Mock Model Data (Update these values based on your actual KMeans results)
# These represent the cluster centers for your 5 variables
centers_data = {
    'ALT': [20.0, 150.0, 45.0],
    'GGT': [30.0, 200.0, 80.0],
    'non_HDL_C_mmol': [3.1, 5.5, 4.2],
    'TyG': [8.5, 10.2, 9.1],
    'bmi': [21.6, 24.5, 23.0]
}
centers_df = pd.DataFrame(centers_data)

# 3. Sidebar: Input Sample Features
st.sidebar.header("Input Sample Features")
alt = st.sidebar.slider("ALT", 1.0, 674.0, 20.0)
ggt = st.sidebar.slider("GGT", 4.0, 652.0, 30.0)
non_hdl = st.sidebar.slider("non_HDL_C_mmol", 0.63, 15.47, 3.16)
tyg = st.sidebar.slider("TyG", 6.9, 11.7, 8.5)
bmi = st.sidebar.slider("bmi", 14.5, 25.0, 21.6)

# 4. Prediction Logic
if st.sidebar.button("Predict Cluster", type="primary"):
    new_sample = np.array([alt, ggt, non_hdl, tyg, bmi])

    # Calculate Euclidean Distance
    dists = np.sqrt(((centers_df.values - new_sample) ** 2).sum(axis=1))
    cluster_id = np.argmin(dists) + 1

    # 5. Results Display
    st.title(f"Prediction Result: Cluster {cluster_id}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distance Analysis")
        # Creating a English version of the distance dataframe
        dist_df = pd.DataFrame({
            'Cluster Group': [f"Cluster {i + 1}" for i in range(len(dists))],
            'Euclidean Distance': dists
        })
        st.bar_chart(dist_df.set_index('Cluster Group'))

    with col2:
        st.subheader("Comparison with Cluster Centers")
        fig, ax = plt.subplots()
        plot_df = centers_df.copy()
        plot_df['Cluster'] = [f"Cluster {i + 1}" for i in range(len(plot_df))]
        long_df = plot_df.melt(id_vars='Cluster')

        # Plotting using English labels
        sns.barplot(data=long_df, x='variable', y='value', hue='Cluster')
        plt.xlabel("Variable")
        plt.ylabel("Value")
        st.pyplot(fig)
else:
    st.info("Please adjust the parameters on the left and click the 'Predict Cluster' button.")