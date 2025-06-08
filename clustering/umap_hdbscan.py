"""
Module for dimensionality reduction and clustering using UMAP and HDBSCAN.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import umap
import hdbscan
from sklearn.metrics import silhouette_score
import logging
import json

from config import UMAP_CONFIG, HDBSCAN_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClusterAnalyzer:
    def __init__(self, use_umap: bool = True):
        """
        Initialize the cluster analyzer.
        
        Args:
            use_umap (bool): Whether to use UMAP for dimensionality reduction before clustering
        """
        self.use_umap = use_umap
        
        if use_umap:
            # Log UMAP configuration
            logger.info("\n=== UMAP Configuration ===")
            logger.info("Parameters:")
            for param, value in UMAP_CONFIG.items():
                logger.info(f"  {param}: {value}")
            logger.info("\nUMAP Parameter Descriptions:")
            logger.info("  n_neighbors: Controls local neighborhood size for manifold approximation")
            logger.info("  n_components: Number of dimensions to reduce to")
            logger.info("  metric: Distance metric for computing distances")
            logger.info("  random_state: Seed for reproducibility")
            
            self.reducer = umap.UMAP(**UMAP_CONFIG)
        
        # Log HDBSCAN configuration
        logger.info("\n=== HDBSCAN Configuration ===")
        logger.info("Parameters:")
        for param, value in HDBSCAN_CONFIG.items():
            logger.info(f"  {param}: {value}")
        logger.info("\nHDBSCAN Parameter Descriptions:")
        logger.info("  min_cluster_size: Minimum size of clusters")
        logger.info("  min_samples: Number of samples in neighborhood for core points")
        logger.info("  metric: Distance metric for computing distances")
        
        self.clusterer = hdbscan.HDBSCAN(**HDBSCAN_CONFIG)
        logger.info("\nInitialized ClusterAnalyzer with " + 
                   ("UMAP and " if use_umap else "") + "HDBSCAN")
    
    def reduce_and_cluster(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reduce dimensionality (if enabled) and perform clustering.
        
        Args:
            embeddings (np.ndarray): High-dimensional embeddings
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - If use_umap=True: (UMAP projections, cluster labels)
                - If use_umap=False: (original embeddings, cluster labels)
        """
        # Log input data shape
        logger.info(f"\nInput embeddings shape: {embeddings.shape}")
        
        if self.use_umap:
            # Dimensionality reduction
            logger.info("\n=== Performing UMAP Dimensionality Reduction ===")
            reduced_data = self.reducer.fit_transform(embeddings)
            logger.info(f"UMAP output shape: {reduced_data.shape}")
            
            # Log UMAP performance metrics
            logger.info("\nUMAP Performance Metrics:")
            logger.info(f"  Number of points: {len(reduced_data)}")
            logger.info(f"  Output dimensions: {reduced_data.shape[1]}")
            logger.info(f"  Dimension reduction ratio: {embeddings.shape[1] / reduced_data.shape[1]:.1f}x")
        else:
            reduced_data = embeddings
            logger.info("\n=== Performing Direct Clustering on High-Dimensional Space ===")
            logger.info(f"  Number of points: {len(embeddings)}")
            logger.info(f"  Input dimensions: {embeddings.shape[1]}")
        
        # Clustering
        logger.info("\n=== Performing HDBSCAN Clustering ===")
        cluster_labels = self.clusterer.fit_predict(reduced_data)
        
        # Calculate and log clustering metrics
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        cluster_sizes = np.bincount(cluster_labels[cluster_labels >= 0])
        
        logger.info("\nClustering Results:")
        logger.info(f"  Number of clusters: {n_clusters}")
        logger.info(f"  Number of noise points: {n_noise}")
        logger.info(f"  Percentage of noise points: {(n_noise/len(cluster_labels))*100:.2f}%")
        logger.info("\nCluster Size Distribution:")
        logger.info(f"  Minimum cluster size: {min(cluster_sizes)}")
        logger.info(f"  Maximum cluster size: {max(cluster_sizes)}")
        logger.info(f"  Average cluster size: {np.mean(cluster_sizes):.2f}")
        
        return reduced_data, cluster_labels
    
    def get_cluster_metrics(self, embeddings: np.ndarray, cluster_labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate clustering quality metrics.
        
        Args:
            embeddings (np.ndarray): Original embeddings
            cluster_labels (np.ndarray): Cluster assignments
            
        Returns:
            Dict[str, float]: Dictionary of metric names and values
        """
        # Filter out noise points for silhouette score
        mask = cluster_labels != -1
        if sum(mask) > 1:  # Need at least 2 points for silhouette score
            silhouette = float(silhouette_score(embeddings[mask], cluster_labels[mask]))
        else:
            silhouette = 0.0
        
        # Calculate additional metrics
        n_clusters = int(len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0))
        n_noise = int(list(cluster_labels).count(-1))
        cluster_sizes = np.bincount(cluster_labels[cluster_labels >= 0])
        
        metrics = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_percentage': float((n_noise/len(cluster_labels))*100),
            'min_cluster_size': int(min(cluster_sizes)),
            'max_cluster_size': int(max(cluster_sizes)),
            'avg_cluster_size': float(np.mean(cluster_sizes)),
            'silhouette_score': silhouette
        }
        
        # Log detailed metrics
        logger.info("\n=== Detailed Clustering Metrics ===")
        logger.info(json.dumps(metrics, indent=2))
        
        return metrics

def reduce_and_cluster(embeddings: np.ndarray, use_umap: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to reduce dimensionality and perform clustering.
    
    Args:
        embeddings (np.ndarray): High-dimensional embeddings
        use_umap (bool): Whether to use UMAP for dimensionality reduction
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - If use_umap=True: (UMAP projections, cluster labels)
            - If use_umap=False: (original embeddings, cluster labels)
    """
    analyzer = ClusterAnalyzer(use_umap=use_umap)
    return analyzer.reduce_and_cluster(embeddings) 