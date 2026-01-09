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
            transform=plt.gca().transAxes,
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
    width = -0.0075 * np.min([np.subtract(*xlim), np.subtract(*ylim)])

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
