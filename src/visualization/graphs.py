import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib
import seaborn as sns


def plot_continuous_columns(
    dataset: pd.DataFrame,
    continuous_columns: list[str],  
) -> matplotlib.figure.Figure:
    
    """
    Generates a set of plots (distribution, histogram, and box plot) for each continuous column in a dataset.

    Parameters
    ----------
    dataset : pd.DataFrame
        The dataset containing the continuous columns to be visualized.
    continuous_columns : list[str]
        A list of column names in the dataset that contain continuous data. Each column in this list
        will have its own row of visualizations, including a distribution plot, histogram, and box plot.

    Returns
    -------
    `~matplotlib.figure.Figure`
    """
    
    fig = plt.figure(figsize=(20, 4 * len(continuous_columns)))
    ax = fig.subplots(nrows=len(continuous_columns), ncols=3)
    for idx, col in enumerate(continuous_columns):
        ax[idx][0].set_ylabel(col, fontsize='x-large')
        ax[idx][1].set_ylabel(' ')
        ax[idx][2].set_ylabel(' ')

        ax[idx][0].set_xlabel('Distrubition')
        ax[idx][1].set_xlabel('Histogram')
        ax[idx][2].set_xlabel('Box Plot')

        sns.kdeplot(dataset[col], ax=ax[idx][0])
        sns.histplot(dataset[col], ax=ax[idx][1])
        sns.boxplot(dataset[col], ax=ax[idx][2])

    return fig


def plot_corelation_matrix(
    dataset: pd.DataFrame,
    continuous_columns: list[str], 
) -> matplotlib.figure.Figure:
    
    correlation_matrix = dataset[continuous_columns].corr()

    fig = plt.figure(figsize=(15, 7))
    ax = fig.subplots()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title('Correlation Map')

    return fig


def pca_visualization(
    pca_results: list[np.ndarray],
    labels: list[str],
    alpha: float = 0.3,
    projection_3d = False,
    cmap = None
) -> matplotlib.figure.Figure:

    fig = plt.figure()
    if projection_3d:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    
    for pca_result, label in zip(pca_results, labels):
        if projection_3d:
            ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], label=label, alpha=alpha)
        else:    
            ax.scatter(pca_result[:, 0], pca_result[:, 1], label=label, alpha=alpha)
    
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    
    if projection_3d:
        ax.set_zlabel('PCA Component 3')
    
    ax.legend()
    ax.set_title(f'{"3D" if projection_3d else "2D"} PCA Visualization')

    fig.tight_layout()
    return fig


def tsne_visualization(
    tsne_results: list[np.ndarray],
    labels: list[str],
    alpha: float = 0.3,
    projection_3d = False,
    cmap = None
) -> matplotlib.figure.Figure:

    fig = plt.figure()
    if projection_3d:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    
    for tsne_result, label in zip(tsne_results, labels):
        if projection_3d:
            ax.scatter(tsne_result[:, 0], tsne_result[:, 1], tsne_result[:, 2], label=label, alpha=alpha)
        else:    
            ax.scatter(tsne_result[:, 0], tsne_result[:, 1], label=label, alpha=alpha)
    
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    
    if projection_3d:
        ax.set_zlabel('t-SNE Component 3')
    
    ax.legend()
    ax.set_title(f'{"3D" if projection_3d else "2D"} t-SNE Visualization')

    fig.tight_layout()
    return fig
    
    