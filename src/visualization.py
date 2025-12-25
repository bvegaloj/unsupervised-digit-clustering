# Libraries
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

def plot_clusters(
    cluster_centers: np.ndarray, 
    rows: int=2, 
    figsize: Tuple[int,int]=(15, 9), 
    cmap: str='binary'
) -> Tuple[plt.Figure, np.ndarray]:
    """Visualize cluster centers as 28x28 digit images

    Args:
        cluster_centers (np.ndarray): Cluster center vectors to visualize
        rows (int, optional): Number of rows in the subplot grid. Defaults=2.
        figsize (tuple of int, optional): Figure size (width, height) in inches. Defaults to (15, 9).
        cmap (str, optional): Matplotlib colormap name. Default='binary'
        
    Returns:
        fig (matplotlib.figure.Figure): Figure object
        axes (np.darray): Array of axes objects
    """
    fig, axes = plt.subplots(rows, 5, figsize=figsize)
    
    for i in range(len(cluster_centers)):
        cluster = cluster_centers[i].reshape(28,-1) * 255
        axes[i//5][i%5].imshow(cluster, cmap=cmap)
        axes[i//5][i%5].set_title(f'Cluster {i + 1}')
    
    plt.tight_layout()
    
    return fig, axes

def plot_digits(
    images: np.ndarray,
    labels: Optional[np.ndarray] = None,
    n_samples: int = 10,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = 'binary'
) -> Tuple[plt.Figure, np.ndarray]:
    """Display grid of digit images

    Args:
        images (np.ndarray): shape (n_images, 784) or (n_images, 28, 28)
        labels (np.ndarray, optional): Labels to display as titles for each image. Defaults to None.
        n_samples (int, optional): Number of images to display. Defaults to 10.
        figsize (tuple of float, optional): Custom figsize. Defaults to None.
        cmap (str, optional): Matplotlib colormap name. Default='binary'
    
    Returns:
        fig (matplotlib.figure.Figure): Figure object
        axes (np.ndarray): Array of axes object
    """
    n_samples = min(n_samples, len(images))
    
    if figsize is None:
        figsize = (n_samples * 1.5, 2)
        
    fig, axes = plt.subplots(1, n_samples, figsize=figsize)
    
    if n_samples == 1:
        axes = [axes]
    
    for i in range(n_samples):
        img = images[i].reshape(28,28)
        
        axes[i].imshow(img, cmap=cmap)
        axes[i].axis('off')
        
        if labels is not None:
            axes[i].set_title(f'{labels[i]}')
        
    plt.tight_layout()
    
    return fig, axes

def plot_single_digit(
    image: np.ndarray,
    label: Optional[int] = None,
    cmap: str = 'binary',
    figsize: Tuple[float, float] = (3, 3)
) -> Tuple[plt.Figure, plt.Axes]:
    """Display a single digit image

    Args:
        image (np.ndarray): shape (784,) or (28x28. Single image to display
        label (int, optional): Label to display as title. Defaults to None.
        cmap (str, optional): Matplotlib colormap name. Defaults to 'binary'.
        figsize (tuple of float): Figure size (width, height) in inches: Defaults to (3,3)

    Returns:
        fig (matplotlib.figure.Figure): Figure object
        axes (np.ndarray): Array of axes object
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    img = image.reshape(28, 28)
    
    ax.imshow(img, cmap=cmap)
    ax.axis('off')
    
    if label is not None:
        ax.set_title(f'Label: {label}')
    
    plt.tight_layout()
    return fig, ax