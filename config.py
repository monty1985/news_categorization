"""
Flexible configuration for text/document clustering project.
"""

CONFIG = {
    'data': {
        'type': 'huggingface',  # 'huggingface', 'csv', 'json', 'txt', etc.
        'source': 'ag_news',   # dataset name or file path
        'split': 'train[:2000]',
        'text_field': 'text',
        'label_field': 'label',
        'preprocessing': {
            'remove_urls': True,
            'remove_special_chars': True,
            'lowercase': True,
            'remove_stopwords': False,
            'lemmatize': False
        }
    },
    'model': {
        'type': 'sentence_transformer',  # or 'word2vec', 'doc2vec', etc.
        'model_name': 'intfloat/e5-base-v2',
        'parameters': {
            'max_tokens': 512,
            'batch_size': 32,
            'pooling': 'mean',
            'normalize': True,
            'truncation': True,
            'truncation_strategy': 'longest_first'
        }
    },
    'clustering': {
        'use_umap': True,
        'algorithm': 'hdbscan',  # 'hdbscan', 'kmeans', 'dbscan', 'hierarchical'
        'umap': {
            'n_neighbors': 30,          # Increased to capture more global structure
            'n_components': 3,
            'metric': 'cosine',
            'random_state': 42,
            'min_dist': 0.0,            # Allow points to be closer together
            'spread': 1.0,              # Control the spread of points
            'local_connectivity': 1.0    # Ensure local connectivity
        },
        'hdbscan': {
            'min_cluster_size': 20,     # Small enough to capture smaller categories
            'min_samples': 5,           # Small enough to allow cluster formation
            'metric': 'euclidean',
            'cluster_selection_method': 'eom',  # Changed back to 'eom' for natural clustering
            'allow_single_cluster': True,
            'prediction_data': True,
            'alpha': 1.0,               # Increased to be more conservative in merging
            'cluster_selection_epsilon': 0.0,    # Disabled epsilon to use full hierarchy
            'algorithm': 'best',         # Use best algorithm for stability
            'core_dist_n_jobs': 1       # Single thread for stability
        },
        'kmeans': {
            'n_clusters': 4,            # Number of clusters
            'init': 'k-means++',        # Initialization method
            'n_init': 10,               # Number of times to run with different centroid seeds
            'max_iter': 300,            # Maximum number of iterations
            'random_state': 42,         # Random state for reproducibility
            'algorithm': 'lloyd'         # Algorithm to use ('lloyd' or 'elkan')
        },
        'dbscan': {
            'eps': 0.5,                 # Maximum distance between samples
            'min_samples': 5,           # Minimum number of samples in a neighborhood
            'metric': 'euclidean',      # Metric to use
            'algorithm': 'auto',        # Algorithm to use ('auto', 'ball_tree', 'kd_tree', 'brute')
            'leaf_size': 30,            # Leaf size for tree-based algorithms
            'p': None                   # Power parameter for Minkowski metric
        },
        'hierarchical': {
            'n_clusters': 4,            # Number of clusters
            'linkage': 'ward',          # Linkage criterion ('ward', 'complete', 'average', 'single')
            'metric': 'euclidean',      # Metric to use
            'compute_full_tree': 'auto' # Whether to compute full tree
        }
    },
    'visualization': {
        'figure_size': (12, 8),
        'point_size': 20,
        'color_palette': 'Spectral',
        'save_path': 'plots/',
        'plot_2d': True,
        'plot_3d': True,
        'interactive': True,
        'marker_size': 5,
        'opacity': 0.7,
        'plot_titles': {
            '2d': 'UMAP Projection with Clusters (2D)',
            '3d': 'Interactive 3D UMAP Projection with Clusters'
        }
    },
    'paths': {
        'data_dir': 'data/',
        'embeddings_dir': 'embeddings/',
        'plots_dir': 'plots/',
        'models_dir': 'models/'
    }
} 