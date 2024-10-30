import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Literal

def pca_visualization(
    pca_results: list[np.ndarray],
    labels: list[str],
    alpha: float = 0.3,
    projection_3d = False,
    cmap = None
) -> None:

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

    plt.tight_layout()
    plt.show()



def tsne_visualization(
    tsne_results: list[np.ndarray],
    labels: list[str],
    alpha: float = 0.3,
    projection_3d = False,
    cmap = None
) -> None:

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

    plt.tight_layout()
    plt.show()
    
    
        
    