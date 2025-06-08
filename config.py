"""
Configuration parameters for the news clustering project.
"""

# Dataset configuration
DATASET_CONFIG = {
    'name': 'ag_news',
    'split': 'train[:2000]',
    'text_field': 'text'
}

# Model configuration
MODEL_CONFIG = {
    'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
    'max_tokens': 512,
    'batch_size': 32
}

# UMAP configuration
UMAP_CONFIG = {
    'n_neighbors': 15,
    'n_components': 3,
    'metric': 'cosine',
    'random_state': 42
}

# HDBSCAN configuration
HDBSCAN_CONFIG = {
    'min_cluster_size': 15,
    'min_samples': 5,
    'metric': 'euclidean'
}

# Visualization configuration
VIZ_CONFIG = {
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
        '2d': 'UMAP Projection with HDBSCAN Clusters (2D)',
        '3d': 'Interactive 3D UMAP Projection with HDBSCAN Clusters'
    }
}

# File paths
PATHS = {
    'data_dir': 'data/',
    'embeddings_dir': 'embeddings/',
    'plots_dir': 'plots/',
    'models_dir': 'models/'
} 