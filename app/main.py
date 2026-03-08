import streamlit as st
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Ensure scripts can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_synthetic_drift import create_synthetic_drift
from analysis_methods.drift_component_analysis import DriftComponentAnalysis
from analysis_methods.visualization_utils import create_scatter_plot, add_biplot_arrows, add_drift_info_to_plot
import patchworklib as pw

# Page Config
st.set_page_config(page_title="Drift PCA Configurator", layout="wide")
st.title("Drift PCA Synthetic Data Configurator")

st.sidebar.header("Dataset Parameters")

# Sidebar Controls
num_features = st.sidebar.number_input("Number of Features", min_value=2, max_value=10, value=2)
n_samples = st.sidebar.number_input("Total Number of Samples", min_value=100, max_value=20000, value=4000, step=100)

st.sidebar.subheader("Pre-Drift Parameters")
means_pre = []
stds_pre = []
for i in range(num_features):
    col1, col2 = st.sidebar.columns(2)
    with col1:
        m = st.number_input(f"Feature {i} Mean", value=0.0, key=f"pre_m_{i}")
        means_pre.append(m)
    with col2:
        s = st.number_input(f"Feature {i} Std", value=1.0, min_value=0.01, key=f"pre_s_{i}")
        stds_pre.append(s)

st.sidebar.subheader("Post-Drift Parameters")
means_post = []
stds_post = []
for i in range(num_features):
    col1, col2 = st.sidebar.columns(2)
    with col1:
        m = st.number_input(f"Feature {i} Mean", value=1.0 if i==0 else 0.0, key=f"post_m_{i}")
        means_post.append(m)
    with col2:
        s = st.number_input(f"Feature {i} Std", value=1.0, min_value=0.01, key=f"post_s_{i}")
        stds_post.append(s)

# Grouped toggle for Vectors & Anchor Point
st.sidebar.subheader("Visualization Options")
show_drift_info = st.sidebar.checkbox("Visualize Drift Vectors & Anchor Point", value=True)

if st.sidebar.button("Generate & Visualize", type="primary"):
    with st.spinner("Generating dataset and computing Drift PCA..."):
        # 1. Generate Data (in-memory)
        X_pre, y_pre, X_post, y_post, df_combined = create_synthetic_drift(
            name="streamlit_temp",
            means_pre=means_pre,
            stds_pre=stds_pre,
            means_post=means_post,
            stds_post=stds_post,
            n_samples=n_samples,
            save_to_disk=False  # New parameter we will add to the script
        )
        
        feature_names = [str(i) for i in range(num_features)]

        # 2. Fit Drift PCA
        dca = DriftComponentAnalysis(n_components=2, add_anchor_point=show_drift_info)
        dca.fit(X_pre, X_post)
        
        X_dca_pre_vis = dca.transform(X_pre)
        X_dca_post_vis = dca.transform(X_post)

        # 3. Visualization configuration
        dca_all = np.vstack([X_dca_pre_vis, X_dca_post_vis])
        x_min_d, x_max_d = dca_all[:, 0].min(), dca_all[:, 0].max()
        y_min_d, y_max_d = dca_all[:, 1].min(), dca_all[:, 1].max()

        # Pre Drift Plot
        fig_d1 = create_scatter_plot(
            X_dca_pre_vis, y_pre, "D1", "D2", 
            "Drift PCA on Pre-Drift Data", 0.0, show_time=False
        )
        fig_d1.axvline(0, color='k', linestyle='--')
        fig_d1.axhline(0, color='k', linestyle='--')
        fig_d1.set_xlim(x_min_d - 0.1 * (x_max_d - x_min_d), x_max_d + 0.1 * (x_max_d - x_min_d))
        fig_d1.set_ylim(y_min_d - 0.1 * (y_max_d - y_min_d), y_max_d + 0.1 * (y_max_d - y_min_d))
        add_biplot_arrows(fig_d1, dca.pca, X_dca_pre_vis, feature_names=feature_names)
        add_drift_info_to_plot(fig_d1, dca, show_drift_info, show_drift_info)

        # Post Drift Plot
        fig_d2 = create_scatter_plot(
            X_dca_post_vis, y_post, "D1", "D2", 
            "Drift PCA on Post-Drift Data", 0.0, show_time=False
        )
        fig_d2.axvline(0, color='k', linestyle='--')
        fig_d2.axhline(0, color='k', linestyle='--')
        fig_d2.set_xlim(x_min_d - 0.1 * (x_max_d - x_min_d), x_max_d + 0.1 * (x_max_d - x_min_d))
        fig_d2.set_ylim(y_min_d - 0.1 * (y_max_d - y_min_d), y_max_d + 0.1 * (y_max_d - y_min_d))
        add_biplot_arrows(fig_d2, dca.pca, X_dca_post_vis, feature_names=feature_names)
        add_drift_info_to_plot(fig_d2, dca, show_drift_info, show_drift_info)

        # Combined Plot
        y_combined = np.array(["Pre-Drift"] * len(X_dca_pre_vis) + ["Post-Drift"] * len(X_dca_post_vis))
        
        fig_d3 = create_scatter_plot(
            dca_all, y_combined, "D1", "D2", 
            "Drift PCA Combined Pre/Post Data", 0.0, show_time=False
        )
        fig_d3.axvline(0, color='k', linestyle='--')
        fig_d3.axhline(0, color='k', linestyle='--')
        fig_d3.set_xlim(x_min_d - 0.1 * (x_max_d - x_min_d), x_max_d + 0.1 * (x_max_d - x_min_d))
        fig_d3.set_ylim(y_min_d - 0.1 * (y_max_d - y_min_d), y_max_d + 0.1 * (y_max_d - y_min_d))
        add_biplot_arrows(fig_d3, dca.pca, dca_all, feature_names=feature_names)
        add_drift_info_to_plot(fig_d3, dca, show_drift_info, show_drift_info)

        # Combine into patchworklib figure
        # Stacked: 2 plots on top row, combined plot spanning bottom row
        final_pw = (fig_d1 | fig_d2) / fig_d3
        
        st.success("Analysis Complete!")
        import io
        buf = io.BytesIO()
        final_pw.savefig(buf, format="png", dpi=120)
        buf.seek(0)
        
        # Display the image buffer
        st.image(buf, width='stretch')
