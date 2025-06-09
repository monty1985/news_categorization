"""
Module for dimensionality reduction and clustering using various algorithms.
"""

import numpy as np
import umap
import hdbscan
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import logging
from typing import Tuple, Dict, Any
import json
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

from config import CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClusterAnalyzer:
    def __init__(self, use_umap: bool = True):
        """
        Initialize the cluster analyzer.
        
        Args:
            use_umap (bool): Whether to use UMAP for dimensionality reduction
        """
        self.use_umap = use_umap
        self.umap_reducer = umap.UMAP(**CONFIG['clustering']['umap'])
        
        # Initialize the appropriate clustering algorithm
        algorithm = CONFIG['clustering']['algorithm']
        if algorithm == 'hdbscan':
            self.clusterer = hdbscan.HDBSCAN(**CONFIG['clustering']['hdbscan'])
        elif algorithm == 'kmeans':
            self.clusterer = KMeans(**CONFIG['clustering']['kmeans'])
        elif algorithm == 'dbscan':
            self.clusterer = DBSCAN(**CONFIG['clustering']['dbscan'])
        elif algorithm == 'hierarchical':
            self.clusterer = AgglomerativeClustering(**CONFIG['clustering']['hierarchical'])
        else:
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
        
        # Log configurations
        logger.info("\n=== UMAP Configuration ===")
        for param, value in CONFIG['clustering']['umap'].items():
            logger.info(f"{param}: {value}")
        
        logger.info(f"\n=== {algorithm.upper()} Configuration ===")
        algorithm_config = CONFIG['clustering'][algorithm]
        for param, value in algorithm_config.items():
            logger.info(f"{param}: {value}")
    
    def reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Reduce dimensions using UMAP.
        
        Args:
            embeddings (np.ndarray): High-dimensional embeddings
            
        Returns:
            np.ndarray: Reduced-dimensional embeddings
        """
        logger.info(f"Reducing dimensions from {embeddings.shape[1]} to {CONFIG['clustering']['umap']['n_components']}")
        return self.umap_reducer.fit_transform(embeddings)
    
    def cluster(self, data: np.ndarray) -> np.ndarray:
        """
        Perform clustering on the data.
        
        Args:
            data (np.ndarray): Input data (either original or reduced dimensions)
            
        Returns:
            np.ndarray: Cluster labels
        """
        algorithm = CONFIG['clustering']['algorithm']
        logger.info(f"Performing {algorithm} clustering")
        
        if algorithm == 'hdbscan':
            self.clusterer.fit(data)
            return self.clusterer.labels_
        else:
            return self.clusterer.fit_predict(data)
    
    def get_cluster_metrics(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate clustering metrics.
        
        Args:
            embeddings (np.ndarray): Original embeddings
            labels (np.ndarray): Cluster labels
            
        Returns:
            Dict[str, float]: Dictionary of metric names and values
        """
        metrics = {}
        
        # Calculate silhouette score
        if len(np.unique(labels)) > 1:
            metrics['silhouette_score'] = float(silhouette_score(embeddings, labels))
        
        # Calculate cluster statistics
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = np.sum(labels == -1)
        
        metrics.update({
            'n_clusters': int(n_clusters),
            'n_noise_points': int(n_noise),
            'noise_percentage': float(n_noise / len(labels) * 100)
        })
        
        # Calculate cluster sizes
        cluster_sizes = [len(np.where(labels == i)[0]) for i in unique_labels if i != -1]
        if cluster_sizes:
            metrics.update({
                'min_cluster_size': int(min(cluster_sizes)),
                'max_cluster_size': int(max(cluster_sizes)),
                'avg_cluster_size': float(np.mean(cluster_sizes))
            })
        
        return metrics

def reduce_and_cluster(embeddings: np.ndarray, use_umap: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce dimensions and perform clustering.
    
    Args:
        embeddings (np.ndarray): High-dimensional embeddings
        use_umap (bool): Whether to use UMAP for dimensionality reduction
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Reduced dimensions and cluster labels
    """
    analyzer = ClusterAnalyzer(use_umap=use_umap)
    
    if use_umap:
        reduced_data = analyzer.reduce_dimensions(embeddings)
    else:
        reduced_data = embeddings
    
    cluster_labels = analyzer.cluster(reduced_data)
    
    return reduced_data, cluster_labels 