import os
import sys
import glob
import json
import logging
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Setup path so imports from analysis_methods and scripts work
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from analysis_methods.drift_component_analysis import DriftComponentAnalysis
from analysis_methods.dca_utils import (
    plot_dca_scatter,
    plot_loadings_compass,
    plot_drift_compass
)
from scripts.run_dca import load_and_scale_data
from scripts.dataset_generation.generate_synthetic_drift import create_synthetic_drift_classes

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="DriftPCA - Concept Drift Visualizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics and styling
st.markdown("""
<style>
    /* Styling headers */
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #1f78b4;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-family: 'Inter', sans-serif;
        color: #555555;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    /* Section Cards */
    .card {
        background-color: #f9f9f9;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=True)
def get_dca_visualization(params_json):
    """
    Fits and generates the DriftPCA figures, caching the result based on a serialized JSON parameter string.
    This prevents recalculation when the parameters are not updated.
    """
    params = json.loads(params_json)
    
    pre_file = os.path.join(params['data_dir_abs'], f"{params['dataset_name']}_pre.csv")
    post_file = os.path.join(params['data_dir_abs'], f"{params['dataset_name']}_post.csv")
    
    if not os.path.exists(pre_file) or not os.path.exists(post_file):
        raise FileNotFoundError(f"Missing dataset files:\n- {pre_file}\n- {post_file}")
        
    # Load and scale data
    X_pre, y_pre, X_post, y_post, feature_names, classes = load_and_scale_data(
        dataset_name=params['dataset_name'],
        data_dir=params['data_dir_abs'],
        ignore_classes=(params['drift_mode'] == "data"),
        has_target=(not params['no_target']),
        pre_path=pre_file,
        post_path=post_file
    )
    
    # Validate boundary and model choices
    needs_model = params['draw_boundary'] or params['highlight_misclassifications'] or params['feature_importance']
    draw_b = params['draw_boundary']
    highlight_mc = params['highlight_misclassifications']
    feat_imp = params['feature_importance']
    
    if needs_model and len(np.unique(y_pre)) < 2:
        draw_b = False
        highlight_mc = False
        feat_imp = False
        needs_model = False
        
    # Train model on Pre-drift data ONLY
    pre_drift_model = None
    if needs_model:
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.inspection import permutation_importance
        
        if params['model_choice'] == "svc":
            pre_drift_model = SVC(kernel='rbf', probability=True, random_state=42)
        else:
            pre_drift_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        
        pre_drift_model.fit(X_pre, y_pre)
        
    feature_importances = None
    if feat_imp and pre_drift_model is not None:
        from sklearn.inspection import permutation_importance
        result = permutation_importance(pre_drift_model, X_pre, y_pre, n_repeats=5, random_state=42, n_jobs=-1)
        feature_importances = np.maximum(result.importances_mean, 0)
        
    # Fit Drift Component Analysis
    by_class = (params['drift_mode'] == "per-class")
    dca = DriftComponentAnalysis(n_components=2, by_class=by_class)
    dca.fit(X_pre, X_post, y_pre, y_post)
    
    # Create unified visualization figure
    fig = plt.figure(figsize=(14, 14))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.5, 1])
    
    ax_scatter = fig.add_subplot(gs[0, :])     
    ax_loadings = fig.add_subplot(gs[1, 0])     
    ax_drift = fig.add_subplot(gs[1, 1])        
    
    contour_info = plot_dca_scatter(
        X_pre, y_pre, X_post, y_post, dca, ax=ax_scatter,
        pre_drift_model=pre_drift_model, color_scheme=params['color_scheme'],
        discrete_boundary=params['discrete_boundary'],
        draw_boundary=draw_b,
        highlight_misclassifications=highlight_mc,
        hide_pre_drift_points=params['hide_pre_drift_points'],
        grid_points=params['grid_points'],
        drift_type=params['drift_type']
    )
    
    if isinstance(contour_info, tuple):
        contour, is_discrete = contour_info
    else:
        contour, is_discrete = contour_info, False
        
    if contour is not None and not is_discrete and len(classes) == 2:
        cbar_ax = fig.add_axes([0.91, 0.55, 0.02, 0.3])
        fig.colorbar(contour, cax=cbar_ax, label=f"Probability of Class {classes[1]}")
        fig.subplots_adjust(right=0.88, hspace=0.3, wspace=0.3)
        
    # Plot 2: Loadings Compass Rose
    plot_loadings_compass(dca, ax=ax_loadings, feature_names=feature_names, scale_loadings=(not params['unscaled_loadings']), feature_importances=feature_importances)
    
    # Plot 3: Drift Compass Rose
    plot_drift_compass(dca, ax=ax_drift, classes=classes if params['drift_mode'] != "data" else None, color_scheme=params['color_scheme'])
    
    if contour is None:
        fig.tight_layout()
        
    return fig


def main():
    st.markdown('<div class="main-title">DriftPCA Showcase</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Interpretable 2D visualizations of concept drift in datasets, models, and explanations.</div>', unsafe_allow_html=True)
    
    # Establish tabs
    tab_vis, tab_gen = st.tabs(["📊 Visualize Concept Drift", "🧪 Generate Synthetic Data"])
    
    # Session state initialization for visualization triggers
    if 'run_dca' not in st.session_state:
        st.session_state.run_dca = False
    if 'run_params' not in st.session_state:
        st.session_state.run_params = None

    # Sidebar - Data Loading
    st.sidebar.header("📁 Data Settings")
    
    # Auto-scan workspace data directories
    pre_files_discovered = glob.glob(os.path.join(PROJECT_ROOT, "data", "**", "*_pre.csv"), recursive=True)
    discovered_dirs = sorted(list(set([
        os.path.dirname(os.path.relpath(f, PROJECT_ROOT)).replace("\\", "/")
        for f in pre_files_discovered
    ])))
    
    enter_path_manually = st.sidebar.checkbox("Type Folder Path Manually", value=False)
    
    if enter_path_manually or not discovered_dirs:
        data_dir = st.sidebar.text_input("Folder Path", "data/synthetic/gen")
    else:
        # Default index should select a folder or synthetic class
        default_idx = discovered_dirs.index("data/synthetic/class") if "data/synthetic/class" in discovered_dirs else 0
        data_dir = st.sidebar.selectbox("Select Discovered Folder", discovered_dirs, index=default_idx)
        
    data_dir_abs = os.path.abspath(os.path.join(PROJECT_ROOT, data_dir))
    
    # Check directory existence and search datasets
    dataset_name = None
    if os.path.exists(data_dir_abs):
        pre_files = glob.glob(os.path.join(data_dir_abs, "*_pre.csv"))
        if pre_files:
            datasets = sorted(list(set([os.path.basename(f).replace("_pre.csv", "") for f in pre_files])))
            dataset_name = st.sidebar.selectbox("Select Dataset", datasets)
        else:
            st.sidebar.warning("No datasets ending in '_pre.csv' found in this directory.")
            dataset_name = st.sidebar.text_input("Manual Dataset Prefix", "rbf")
    else:
        st.sidebar.error("Folder path does not exist.")
        dataset_name = st.sidebar.text_input("Manual Dataset Prefix", "rbf")
        
    # Sidebar - DCA Pipeline Parameters
    st.sidebar.header("⚙️ Method Parameters")
    
    no_target = st.sidebar.checkbox("Unsupervised Mode (No Target/Label)", value=False)
    
    # Enable appropriate drift modes based on target checkbox
    if no_target:
        drift_mode = "data"
        st.sidebar.info("Unsupervised mode enforces 'data' drift mode.")
    else:
        drift_mode = st.sidebar.selectbox("Drift Calculation Mode", ["per-class", "global", "data"], index=0)
        
    model_choice = st.sidebar.selectbox("Pre-drift Model", ["svc", "rf"], index=0)
    
    # Color schemes and highlights
    color_scheme = st.sidebar.selectbox("Points Color Scheme", ["class", "drift"], index=0 if drift_mode != "data" else 1)
    
    # misclassifications and importance flags
    highlight_misclassifications = False
    feature_importance = False
    draw_boundary = True
    
    if drift_mode != "data" and not no_target:
        highlight_misclassifications = st.sidebar.checkbox("Highlight Misclassifications", value=True)
        feature_importance = st.sidebar.checkbox("Compute Feature Importances", value=True)
        draw_boundary = st.sidebar.checkbox("Draw Pre-drift Model Decision Boundary", value=True)
        
    discrete_boundary = False
    if draw_boundary and not no_target and drift_mode != "data":
        discrete_boundary = st.sidebar.checkbox("Discrete (Hard) Decision Boundary", value=False)
        
    unscaled_loadings = st.sidebar.checkbox("Disable SVD loading arrow scaling", value=False)
    drift_type = st.sidebar.selectbox("Concept Drift Time Style", ["sudden", "gradual"], index=0)
    hide_pre_drift_points = st.sidebar.checkbox("Hide Pre-Drift Points", value=False)
    grid_points = st.sidebar.slider("Decision Boundary Grid Density", min_value=20, max_value=300, value=100, step=10)
    
    # primary Run Button in Sidebar to snapshot settings and trigger run
    if st.sidebar.button("Run DCA Visualization", type="primary"):
        st.session_state.run_dca = True
        st.session_state.run_params = {
            'data_dir_abs': data_dir_abs,
            'dataset_name': dataset_name,
            'no_target': no_target,
            'drift_mode': drift_mode,
            'model_choice': model_choice,
            'color_scheme': color_scheme,
            'highlight_misclassifications': highlight_misclassifications,
            'feature_importance': feature_importance,
            'draw_boundary': draw_boundary,
            'discrete_boundary': discrete_boundary,
            'unscaled_loadings': unscaled_loadings,
            'drift_type': drift_type,
            'hide_pre_drift_points': hide_pre_drift_points,
            'grid_points': grid_points
        }
    
    # Visualizer Tab
    with tab_vis:
        st.subheader("📊 DriftPCA Interactive Visualizer")
        
        if not st.session_state.run_dca or st.session_state.run_params is None:
            st.info("👈 Configure data and method parameters in the sidebar, then click **Run DCA Visualization** to display the plots.")
        else:
            params = st.session_state.run_params
            
            try:
                # Serialize params as JSON to serve as stable immutable cache key
                params_json = json.dumps(params)
                
                # Fetch cached figure or generate
                fig = get_dca_visualization(params_json)
                
                # Render figure to Streamlit
                st.pyplot(fig)
                
            except Exception as ex:
                st.error(f"Failed to execute DCA pipeline: {ex}")
                st.exception(ex)
                        
    # Dataset Generator Tab
    with tab_gen:
        st.subheader("🧪 Create Custom Class-Specific Drift Dataset")
        st.write("Configure class-specific means and standard deviations to generate a custom synthetic concept drift CSV pair.")
        
        col_main1, col_main2 = st.columns([1, 3])
        
        with col_main1:
            gen_name = st.text_input("Dataset Name prefix", "custom_drift")
            gen_features = st.slider("Number of Features", min_value=1, max_value=10, value=3)
            gen_samples = st.number_input("Number of Samples", min_value=100, max_value=20000, value=4000, step=100)
            gen_data_dir = st.text_input("Output Data Directory", "data/custom")
            
        st.write("---")
        st.write("### Feature Drift Configuration")
        
        means_pre_c0 = []
        stds_pre_c0 = []
        means_pre_c1 = []
        stds_pre_c1 = []
        means_post_c0 = []
        stds_post_c0 = []
        means_post_c1 = []
        stds_post_c1 = []
        
        for f_idx in range(gen_features):
            st.markdown(f"#### Feature {f_idx}")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.write("**Pre-Drift Class 0**")
                m_pre_c0 = st.number_input(f"Mean (Pre C0, F{f_idx})", value=0.0, key=f"m_pre_c0_{f_idx}")
                s_pre_c0 = st.number_input(f"Std (Pre C0, F{f_idx})", value=1.0, min_value=0.1, key=f"s_pre_c0_{f_idx}")
                
            with col2:
                st.write("**Pre-Drift Class 1**")
                m_pre_c1 = st.number_input(f"Mean (Pre C1, F{f_idx})", value=0.0, key=f"m_pre_c1_{f_idx}")
                s_pre_c1 = st.number_input(f"Std (Pre C1, F{f_idx})", value=1.0, min_value=0.1, key=f"s_pre_c1_{f_idx}")
                
            with col3:
                st.write("**Post-Drift Class 0**")
                m_post_c0 = st.number_input(f"Mean (Post C0, F{f_idx})", value=0.0, key=f"m_post_c0_{f_idx}")
                s_post_c0 = st.number_input(f"Std (Post C0, F{f_idx})", value=1.0, min_value=0.1, key=f"s_post_c0_{f_idx}")
                
            with col4:
                st.write("**Post-Drift Class 1**")
                m_post_c1 = st.number_input(f"Mean (Post C1, F{f_idx})", value=0.0 if f_idx == 0 else 0.0, key=f"m_post_c1_{f_idx}")
                s_post_c1 = st.number_input(f"Std (Post C1, F{f_idx})", value=1.0, min_value=0.1, key=f"s_post_c1_{f_idx}")
                
            means_pre_c0.append(m_pre_c0)
            stds_pre_c0.append(s_pre_c0)
            means_pre_c1.append(m_pre_c1)
            stds_pre_c1.append(s_pre_c1)
            means_post_c0.append(m_post_c0)
            stds_post_c0.append(s_post_c0)
            means_post_c1.append(m_post_c1)
            stds_post_c1.append(s_post_c1)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
        if st.button("Generate Dataset"):
            with st.spinner("Generating custom synthetic dataset..."):
                try:
                    data_dir_abs_gen = os.path.abspath(os.path.join(PROJECT_ROOT, gen_data_dir))
                    vis_dir_abs_gen = os.path.abspath(os.path.join(PROJECT_ROOT, "results", "synthetic", "vis"))
                    
                    X_pre, y_pre, X_post, y_post, df_combined = create_synthetic_drift_classes(
                        name=gen_name,
                        means_pre_c0=means_pre_c0, stds_pre_c0=stds_pre_c0,
                        means_pre_c1=means_pre_c1, stds_pre_c1=stds_pre_c1,
                        means_post_c0=means_post_c0, stds_post_c0=stds_post_c0,
                        means_post_c1=means_post_c1, stds_post_c1=stds_post_c1,
                        n_samples=gen_samples,
                        data_dir=data_dir_abs_gen,
                        vis_dir=vis_dir_abs_gen,
                        save_to_disk=True
                    )
                    
                    st.success(f"Successfully generated custom dataset '{gen_name}' with {gen_features} features and saved CSV files to `{gen_data_dir}/`!")
                    
                    # Suggest selectbox reload
                    st.info("🔄 Note: To see the new dataset in the visualizer tab, please switch folders or select it from the sidebar selectbox.")
                except Exception as ex:
                    st.error(f"Failed to generate dataset: {ex}")


if __name__ == "__main__":
    main()
