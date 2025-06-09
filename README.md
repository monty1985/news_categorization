# News Article Clustering

A flexible and powerful text clustering system that can work with any text-based embeddings. This project implements various clustering algorithms to discover natural groupings in text data.

## Features

- **Flexible Embedding Support**: Works with any text-based embeddings (e.g., sentence transformers, word2vec, doc2vec)
- **Multiple Clustering Algorithms**:
  - HDBSCAN: Hierarchical density-based clustering
  - K-means: Traditional centroid-based clustering
  - DBSCAN: Density-based spatial clustering
  - Hierarchical Clustering: Agglomerative clustering with various linkage methods
- **Dimensionality Reduction**: UMAP for efficient visualization and clustering
- **Interactive Visualizations**: 2D and 3D cluster visualizations
- **Comprehensive Analysis**: Cluster metrics and example texts from each cluster

## Installation

```bash
# Create and activate virtual environment
conda create -n news_cluster_env python=3.12
conda activate news_cluster_env

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Run with default HDBSCAN clustering
python main.py

# Run with K-means
python main.py --algorithm kmeans --n-clusters 4

# Run with DBSCAN
python main.py --algorithm dbscan --eps 0.5 --min-samples 5

# Run with Hierarchical Clustering
python main.py --algorithm hierarchical --n-clusters 4
```

### Command Line Arguments

- `--algorithm`: Choose clustering algorithm
  - Options: 'hdbscan', 'kmeans', 'dbscan', 'hierarchical'
  - Default: 'hdbscan'

- Algorithm-specific parameters:
  - K-means:
    - `--n-clusters`: Number of clusters (default: 4)
  - DBSCAN:
    - `--eps`: Maximum distance between samples (default: 0.5)
    - `--min-samples`: Minimum samples in neighborhood (default: 5)
  - HDBSCAN:
    - `--min-cluster-size`: Minimum cluster size (default: 20)
  - Hierarchical:
    - `--n-clusters`: Number of clusters (default: 4)

## Configuration

The project uses a flexible configuration system (`config.py`) that allows customization of:

1. **Data Source**:
   - HuggingFace datasets
   - Custom text files
   - CSV/JSON files

2. **Embedding Model**:
   - Sentence Transformers
   - Word2Vec
   - Doc2Vec
   - Custom embedding models

3. **Clustering Parameters**:
   - UMAP dimensionality reduction settings
   - Algorithm-specific parameters
   - Visualization options

## Using Custom Embeddings

The system can work with any text-based embeddings. To use custom embeddings:

1. Prepare your embeddings as a numpy array
2. Update the configuration in `config.py`:
   ```python
   CONFIG['model'] = {
       'type': 'custom',
       'embedding_dim': your_embedding_dimension
   }
   ```

## Clustering Algorithms

### HDBSCAN
- Hierarchical version of DBSCAN
- Discovers clusters of varying densities
- No need to specify number of clusters
- Good for complex data structures

### K-means
- Traditional centroid-based clustering
- Requires specifying number of clusters
- Works well with spherical clusters
- Fast and scalable

### DBSCAN
- Density-based spatial clustering
- Discovers clusters of arbitrary shapes
- Can identify noise points
- No need to specify number of clusters

### Hierarchical Clustering
- Creates a hierarchy of clusters
- Can be visualized as a dendrogram
- More computationally expensive
- Good for understanding data structure

## Output

The system provides:

1. **Clustering Metrics**:
   - Number of clusters
   - Silhouette score
   - Cluster sizes
   - Noise points (if applicable)

2. **Visualizations**:
   - 2D UMAP projection
   - 3D interactive visualization
   - Cluster distribution plots

3. **Cluster Analysis**:
   - Example texts from each cluster
   - Cluster statistics
   - Representative samples

## Project Structure

```
news_categorization/
├── config.py              # Configuration settings
├── main.py               # Main execution script
├── requirements.txt      # Project dependencies
├── embeddings/           # Embedding generation
├── clustering/           # Clustering algorithms
├── visualization/        # Plotting utilities
└── data/                 # Data storage
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details. 