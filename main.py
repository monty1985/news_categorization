"""
Main script for news article clustering using various algorithms.
"""

import os
import logging
from datasets import load_dataset
import pandas as pd
from typing import List, Dict, Any, Tuple
import numpy as np
import json
import argparse

from config import CONFIG
from embeddings.embedder import get_embeddings
from clustering.umap_hdbscan import reduce_and_cluster, ClusterAnalyzer
from visualization.plot_clusters import plot_umap_clusters

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='News Article Clustering')
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['hdbscan', 'kmeans', 'dbscan', 'hierarchical'],
        default='hdbscan',
        help='Clustering algorithm to use (default: hdbscan)'
    )
    parser.add_argument(
        '--n-clusters',
        type=int,
        help='Number of clusters for kmeans and hierarchical clustering'
    )
    parser.add_argument(
        '--eps',
        type=float,
        help='Maximum distance between samples for DBSCAN'
    )
    parser.add_argument(
        '--min-samples',
        type=int,
        help='Minimum number of samples in a neighborhood for DBSCAN'
    )
    parser.add_argument(
        '--min-cluster-size',
        type=int,
        help='Minimum cluster size for HDBSCAN'
    )
    return parser.parse_args()

def update_config_from_args(args):
    """
    Update CONFIG with command line arguments.
    
    Args:
        args (argparse.Namespace): Parsed command line arguments
    """
    # Update algorithm
    CONFIG['clustering']['algorithm'] = args.algorithm
    
    # Update algorithm-specific parameters
    if args.algorithm == 'kmeans' and args.n_clusters:
        CONFIG['clustering']['kmeans']['n_clusters'] = args.n_clusters
    elif args.algorithm == 'hierarchical' and args.n_clusters:
        CONFIG['clustering']['hierarchical']['n_clusters'] = args.n_clusters
    elif args.algorithm == 'dbscan':
        if args.eps:
            CONFIG['clustering']['dbscan']['eps'] = args.eps
        if args.min_samples:
            CONFIG['clustering']['dbscan']['min_samples'] = args.min_samples
    elif args.algorithm == 'hdbscan' and args.min_cluster_size:
        CONFIG['clustering']['hdbscan']['min_cluster_size'] = args.min_cluster_size

def load_data() -> List[str]:
    """
    Load the AG News dataset.
    
    Returns:
        List[str]: List of news article texts
    """
    logger.info(f"Loading {CONFIG['data']['source']} dataset")
    dataset = load_dataset(CONFIG['data']['source'], split=CONFIG['data']['split'])
    texts = [x[CONFIG['data']['text_field']] for x in dataset]
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
    # Parse command line arguments
    args = parse_arguments()
    update_config_from_args(args)
    
    # Create necessary directories
    for path in CONFIG['paths'].values():
        os.makedirs(path, exist_ok=True)
    
    # Load data
    texts = load_data()
    logger.info("Loaded Dataset Successfully")
    
    # Generate embeddings
    logger.info("Generating embeddings")
    embeddings = get_embeddings(texts, save=True, filename="ag_news_embeddings.npy")
    
    # Perform clustering with UMAP reduction
    logger.info(f"\n=== Performing Clustering with {args.algorithm.upper()} ===")
    umap_proj, cluster_labels = reduce_and_cluster(embeddings, use_umap=True)
    
    # Log UMAP projection shape
    logger.info(f"UMAP projection shape: {umap_proj.shape}")
    
    # Calculate clustering metrics
    analyzer = ClusterAnalyzer(use_umap=True)
    metrics = analyzer.get_cluster_metrics(embeddings, cluster_labels)
    logger.info(f"Clustering metrics: {metrics}")
    
    # Visualize results
    logger.info("Creating UMAP visualizations")
    plot_umap_clusters(umap_proj, cluster_labels, texts)
    
    # Print detailed metrics
    logger.info("\n=== Clustering Metrics ===")
    logger.info(json.dumps(metrics, indent=2))
    
    # Analyze clusters
    logger.info("\n=== Analyzing Clusters ===")
    cluster_analysis = analyze_clusters(texts, cluster_labels)
    
    # Print cluster summaries
    logger.info("\n=== Cluster Summaries ===")
    for cluster_id, texts in cluster_analysis.items():
        logger.info(f"\nCluster {cluster_id}:")
        for i, text in enumerate(texts, 1):
            logger.info(f"Example {i}: {text[:200]}...")

if __name__ == "__main__":
    main() 