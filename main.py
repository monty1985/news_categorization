"""
Main script for news article clustering using UMAP and HDBSCAN.
"""

import os
import logging
from datasets import load_dataset
import pandas as pd
from typing import List, Dict, Any
import numpy as np
import json

from config import DATASET_CONFIG, PATHS
from embeddings.embedder import get_embeddings
from clustering.umap_hdbscan import reduce_and_cluster, ClusterAnalyzer
from visualization.plot_clusters import plot_umap_clusters

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data() -> List[str]:
    """
    Load the AG News dataset.
    
    Returns:
        List[str]: List of news article texts
    """
    logger.info(f"Loading {DATASET_CONFIG['name']} dataset")
    dataset = load_dataset(DATASET_CONFIG['name'], split=DATASET_CONFIG['split'])
    texts = [x[DATASET_CONFIG['text_field']] for x in dataset]
    logger.info(f"Loaded {len(texts)} articles")
    return texts

def analyze_clusters(texts: List[str], cluster_labels: np.ndarray) -> Dict[int, List[str]]:
    """
    Analyze the contents of each cluster.
    
    Args:
        texts (List[str]): Original texts
        cluster_labels (np.ndarray): Cluster assignments
        
    Returns:
        Dict[int, List[str]]: Dictionary mapping cluster IDs to representative texts
    """
    cluster_texts = {}
    for cluster_id in set(cluster_labels):
        if cluster_id == -1:  # Skip noise points
            continue
        # Convert boolean mask to list indices
        indices = np.where(cluster_labels == cluster_id)[0]
        # Get first 5 texts from each cluster using list comprehension
        cluster_texts[cluster_id] = [texts[i] for i in indices[:5]]
    
    return cluster_texts

def main():
    """Main execution function."""
    # Create necessary directories
    for path in PATHS.values():
        os.makedirs(path, exist_ok=True)
    
    # Load data
    texts = load_data()
    logger.info("Loaded Dataset Successfully")
    
    # Generate embeddings
    logger.info("Generating embeddings")
    embeddings = get_embeddings(texts, save=True, filename="ag_news_embeddings.npy")
    
    # Perform clustering with UMAP reduction
    logger.info("\n=== Performing Clustering with UMAP Reduction ===")
    umap_proj, cluster_labels_umap = reduce_and_cluster(embeddings, use_umap=True)
    
    # Log UMAP projection shape
    logger.info(f"UMAP projection shape: {umap_proj.shape}")
    
    # Calculate clustering metrics for UMAP approach
    analyzer_umap = ClusterAnalyzer(use_umap=True)
    metrics_umap = analyzer_umap.get_cluster_metrics(embeddings, cluster_labels_umap)
    logger.info(f"Clustering metrics with UMAP: {metrics_umap}")
    
    # Visualize UMAP results
    logger.info("Creating UMAP visualizations")
    plot_umap_clusters(umap_proj, cluster_labels_umap, texts)
    
    # Perform direct clustering without UMAP
    logger.info("\n=== Performing Direct Clustering on High-Dimensional Space ===")
    _, cluster_labels_direct = reduce_and_cluster(embeddings, use_umap=False)
    
    # Calculate clustering metrics for direct approach
    analyzer_direct = ClusterAnalyzer(use_umap=False)
    metrics_direct = analyzer_direct.get_cluster_metrics(embeddings, cluster_labels_direct)
    logger.info(f"Clustering metrics without UMAP: {metrics_direct}")
    
    # Compare results
    logger.info("\n=== Comparison of Clustering Approaches ===")
    logger.info("UMAP + Clustering:")
    logger.info(json.dumps(metrics_umap, indent=2))
    logger.info("\nDirect Clustering:")
    logger.info(json.dumps(metrics_direct, indent=2))
    
    # Analyze clusters for both approaches
    logger.info("\n=== Analyzing Clusters from UMAP Approach ===")
    cluster_texts_umap = analyze_clusters(texts, cluster_labels_umap)
    
    logger.info("\n=== Analyzing Clusters from Direct Clustering Approach ===")
    cluster_texts_direct = analyze_clusters(texts, cluster_labels_direct)
    
    # Print cluster summaries for both approaches
    logger.info("\n=== Cluster Summaries: UMAP Approach ===")
    for cluster_id, texts in cluster_texts_umap.items():
        logger.info(f"\nCluster {cluster_id}:")
        for i, text in enumerate(texts, 1):
            logger.info(f"{i}. {text[:100]}...")  # Print first 100 chars of each text
    
    logger.info("\n=== Cluster Summaries: Direct Clustering Approach ===")
    for cluster_id, texts in cluster_texts_direct.items():
        logger.info(f"\nCluster {cluster_id}:")
        for i, text in enumerate(texts, 1):
            logger.info(f"{i}. {text[:100]}...")  # Print first 100 chars of each text

if __name__ == "__main__":
    main() 