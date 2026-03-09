import matplotlib.pyplot as plt
import patchworklib as pw
import seaborn as sns
import pandas as pd
import numpy as np
import logging

# Configure logger
logger = logging.getLogger(__name__)

def create_scatter_plot(
    X_transformed, y, xlabel, ylabel, title, fit_time, show_time, show_title=True
):
    """
    Create and return a scatter plot visualization using seaborn.

    Args:
        X_transformed: 2D numpy array of transformed features
        y: numpy array of target labels
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        title: Title for the plot
        fit_time: Time taken for transformation in seconds
        show_time: Whether to display timing information on the plot

    Returns:
        patchworklib.Brick: The created figure
    """
    # Define NPG color palette (ggsci NPG palette - up to 8 colors)
    npg_palette = [
        "#E64B35",  # Red
        "#4DBBD5",  # Cyan
        "#00A087",  # Teal
        "#3C5488",  # Blue
        "#F39B7F",  # Salmon
        "#8491B4",  # Purple
        "#91D1C2",  # Mint
        "#DC0000",  # Dark Red
    ]

    # Create a DataFrame for seaborn
    plot_df = pd.DataFrame(
        {xlabel: X_transformed[:, 0], ylabel: X_transformed[:, 1], "label": y}
    )

    # Create visualization with larger font
    ax = pw.Brick(figsize=(10, 8))

    # Create scatterplot with seaborn
    sns.scatterplot(
        data=plot_df,
        x=xlabel,
        y=ylabel,
        hue="label",
        palette=npg_palette[: len(np.unique(y))],
        alpha=0.6,
        s=50,
        legend=False,
        ax=ax,
    )

    ax.set(xlabel=xlabel, ylabel=ylabel)
    if show_title:
        ax.set_title(title, fontsize=42)
        y_offset = 0.92
        x_offset = 0.97
    else:
        y_offset = 0.97
        x_offset = 0.97

    # Add timing text if provided
    if show_time:
        ax.text(
            x_offset,
            y_offset,
            f"{fit_time:.5f}s",
            transform=ax.transAxes,
            fontsize=24,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"),
            color="black",
        )

    return ax


def add_biplot_arrows(ax, pca, X_transformed, n_features=None, feature_names=None):
    """
    Add biplot arrows to a PCA plot.

    Args:
        ax: The matplotlib axes object
        pca: Fitted PCA object with components_
        X_transformed: The transformed data (scores)
        n_features: Number of features (if feature_names not provided)
        feature_names: List of feature names for labeling arrows
    """
    # Get loadings (coordinates of features)
    loadings = pca.components_[:2].T

    # Scale arrows based on the range of scores
    arrows = loadings * np.abs(X_transformed[:, :2]).max(axis=0)

    # Calculate arrow width
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    width = -0.0075 * np.min([np.subtract(*xlim), np.subtract(*ylim)]) # TODO fix for scaling issues

    # Draw arrows for each feature
    for i, arrow in enumerate(arrows):
        ax.arrow(
            0,
            0,
            arrow[0],
            arrow[1],
            color="k",
            alpha=0.5,
            width=width,
            ec="none",
            length_includes_head=True,
            zorder=10,
        )

        # Add feature label if provided
        if feature_names is not None:
            label = feature_names[i]
        elif n_features is not None:
            label = f"F{i + 1}"
        else:
            label = f"F{i + 1}"

        ax.text(
            arrow[0] * 1.05,
            arrow[1] * 1.05,
            label,
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"
            ),
            zorder=11,
        )

def add_drift_info_to_plot(ax, dca, add_anchor, add_vectors):
    """
    Add drift vectors (Mean Increase, Std Increase) and Anchor point to the plot.
    Drift vectors are shifted to the PCA space center 0.0 instead of the Anchor point for visual clarity.
    """
    if not add_vectors or dca.diff_vectors is None:
        return
        
    # Transform diff vectors to PCA space
    vectors_trans = dca.transform(dca.diff_vectors)
    
    # Calculate width for arrows based on plot limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # Simple heuristic for width similar to visualization_utils
    width = 0.005 * min(abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0]))

    # start_idx determines if the first diff_vector is the origin anchor
    actually_add_anchor = add_anchor and getattr(dca, 'add_anchor_point', False)
    
    start_idx = 0
    if actually_add_anchor:
        if len(vectors_trans) > 0:
            origin = vectors_trans[0]
            # Plot Data Origin
            ax.scatter(origin[0], origin[1], color='Red', s=100, marker='X', zorder=20, label='Data Origin')
            ax.text(origin[0], origin[1], 'Data Origin', color='Red', fontsize=16, ha='right', va='bottom')
            start_idx = 1
    
    labels = ['Mean Increase', 'Std Increase'] # TODO add for more classes used in PCA fitting
    colors = ['#7802b8', 'green'] # Purple, Green
    
    max_val = np.max(np.abs(vectors_trans))
    min_dimension = min(abs(xlim[1]), abs(ylim[1]))

    # Calculate a single SCALAR scale factor
    # This ensures x and y are multiplied by the same amount, preserving the angle.
    if max_val > 0:
        scale_factor = 0.5 * min_dimension / max_val
    else:
        scale_factor = 1.0

    for i, vec in enumerate(vectors_trans[start_idx:]):
        if i >= len(labels): break

        if actually_add_anchor:
            # Calculate the raw vector difference from the anchor point
            dx = vec[0] - origin[0]
            dy = vec[1] - origin[1]
            # Apply the SCALAR scale_factor to both dx and dy
            ax.arrow(0, 0, dx * scale_factor, dy * scale_factor, 
                        color=colors[i], width=width, length_includes_head=True, zorder=20)
            
            # Label
            ax.text(dx * scale_factor * 1.15, dy * scale_factor * 1.15, 
                    labels[i], color=colors[i], fontsize=16, ha='center', va='center', fontweight='bold', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none"))

        else:
            # Same logic if not using anchor offset (just raw vector)
            ax.arrow(0, 0, vec[0] * scale_factor, vec[1] * scale_factor, 
                        color=colors[i], width=width, length_includes_head=True, zorder=20)
            
            ax.text(vec[0] * scale_factor * 1.15, vec[1] * scale_factor * 1.15, 
                    labels[i], color=colors[i], fontsize=16, ha='center', va='center', fontweight='bold', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none"))

def create_empty_plot(xlabel, ylabel, title):
    """Create an empty patchworklib Brick to draw arrows on."""
    ax = pw.Brick(figsize=(10, 8))
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, fontsize=42)
    return ax

def add_unscaled_biplot_arrows(ax, pca, feature_names=None):
    """Add unscaled raw biplot arrows (loadings) to the plot."""
    loadings = pca.components_[:2].T
    
    # We set limits here dynamically to make sure the arrows fit.
    max_val = np.max(np.abs(loadings)) * 1.5
    if max_val == 0:
        max_val = 1.0
    
    current_xlim = ax.get_xlim()
    current_ylim = ax.get_ylim()
    
    # Only expand limits, don't shrink them if they are already larger
    new_xlim = (min(-max_val, current_xlim[0]), max(max_val, current_xlim[1]))
    new_ylim = (min(-max_val, current_ylim[0]), max(max_val, current_ylim[1]))
    ax.set_xlim(*new_xlim)
    ax.set_ylim(*new_ylim)
    
    width = 0.005 * (new_xlim[1] - new_xlim[0])
    
    for i, arrow in enumerate(loadings):
        ax.arrow(
            0, 0, arrow[0], arrow[1],
            color="k", alpha=0.5, width=width, ec="none",
            length_includes_head=True, zorder=10
        )
        label = feature_names[i] if feature_names is not None else f"F{i + 1}"
        ax.text(
            arrow[0] * 1.1, arrow[1] * 1.1, label,
            ha="center", va="center", fontsize=18, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"),
            zorder=11,
        )

def add_unscaled_drift_info_to_plot(ax, dca, add_vectors):
    """Add unscaled drift vectors acting from origin (0,0)."""
    if not add_vectors or dca.diff_vectors is None:
        return
        
    vectors_trans = dca.transform(dca.diff_vectors)
    
    labels = ['Mean Increase', 'Std Increase']
    colors = ['#7802b8', 'green']
    
    # Start idx based on whether anchor point was generated in diff_vectors
    start_idx = 1 if dca.add_anchor_point else 0
    
    # Update limits dynamically
    max_val = np.max(np.abs(vectors_trans[start_idx:])) * 1.5
    if max_val == 0: max_val = 1.0
    
    current_xlim = ax.get_xlim()
    current_ylim = ax.get_ylim()
    new_xlim = (min(-max_val, current_xlim[0]), max(max_val, current_xlim[1]))
    new_ylim = (min(-max_val, current_ylim[0]), max(max_val, current_ylim[1]))
    ax.set_xlim(*new_xlim)
    ax.set_ylim(*new_ylim)
    
    width = 0.005 * (new_xlim[1] - new_xlim[0])
    
    for i, vec in enumerate(vectors_trans[start_idx:]):
        if i >= len(labels): break
        
        # Draw from 0,0 since SVD/no-anchor implies differences are the vectors directly
        ax.arrow(
            0, 0, vec[0], vec[1],
            color=colors[i], width=width, length_includes_head=True, zorder=20
        )
        
        ax.text(
            vec[0] * 1.15, vec[1] * 1.15, labels[i],
            color=colors[i], fontsize=16, ha='center', va='center', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none")
        )

def plot_decision_boundaries_comparison(X_pre, y_pre, X_post, y_post, 
                                        model_pre, model_post,
                                        pca_model, dca_model,
                                        grid_points=200, use_proba=True):
    """
    Plots decision boundaries in a 2x2 grid comparing PCA and DCA for Pre and Post drift data.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=False, sharex=False)
    
    # Configuration for the 4 subplots
    plot_configs = [
        (axes[0, 0], X_pre, y_pre, pca_model, model_pre, "PCA on Pre-Drift Data (Reference Model)"),
        (axes[0, 1], X_post, y_post, pca_model, model_post, "PCA on Post-Drift Data (New Model)"),
        (axes[1, 0], X_pre, y_pre, dca_model, model_pre, "DCA on Pre-Drift Data (Reference Model)"),
        (axes[1, 1], X_post, y_post, dca_model, model_post, "DCA on Post-Drift Data (New Model)")
    ]
    
    contour = None
    
    for ax, X_data, y_data, projector, current_model, subtitle in plot_configs:
        # Transform data to 2D
        X_proj = projector.transform(X_data)
        
        # Define grid boundaries
        x_min, x_max = X_proj[:, 0].min() - 1, X_proj[:, 0].max() + 1
        y_min, y_max = X_proj[:, 1].min() - 1, X_proj[:, 1].max() + 1
        
        # Create grid
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_points),
                             np.linspace(y_min, y_max, grid_points))
        grid_2d = np.c_[xx.ravel(), yy.ravel()]
        
        # Inverse transform to N-dimensional space
        grid_nd = projector.inverse_transform(grid_2d)
        
        # Predict using the assigned N-dimensional model for this subplot
        if use_proba and hasattr(current_model, "predict_proba"):
            Z = current_model.predict_proba(grid_nd)[:, 1]
            cmap = plt.cm.RdBu
            levels = np.linspace(0, 1, 21)
        else:
            Z = current_model.predict(grid_nd)
            cmap = plt.cm.RdYlBu
            levels = None
            
        Z = Z.reshape(xx.shape)
        
        # Plot contours and data
        if use_proba:
            contour = ax.contourf(xx, yy, Z, levels=levels, alpha=0.8, cmap=cmap, vmin=0, vmax=1)
        else:
            ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)
            
        ax.scatter(X_proj[:, 0], X_proj[:, 1], c=y_data, 
                   s=40, edgecolor='k', cmap=plt.cm.RdYlBu, alpha=0.8)
        
        ax.set_title(subtitle, fontsize=14)
        ax.set_xlabel("Component 1")
        if ax in [axes[0, 0], axes[1, 0]]:
            ax.set_ylabel("Component 2")
            
    if use_proba and contour:
        # Add colorbar globally to the right
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(contour, cax=cbar_ax, label="Probability of Class 1")
        plt.subplots_adjust(right=0.9, hspace=0.3)
    else:
        plt.tight_layout()
        
    return fig