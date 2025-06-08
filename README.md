# News Article Clustering with UMAP and HDBSCAN

This project performs unsupervised topic discovery on news articles using the AG News dataset. It uses sentence transformers to create embeddings, UMAP for dimensionality reduction, and HDBSCAN for clustering.

## Features

- Text embedding using sentence-transformers
- Dimensionality reduction with UMAP
- Clustering with HDBSCAN
- Interactive visualizations with Plotly
- Cluster analysis and interpretation
- Comprehensive logging and metrics

## Project Structure

```
agnews-umap-hdbscan-clustering/
│
├── data/                    # Data storage
├── embeddings/             # Embedding generation and storage
├── clustering/             # UMAP and HDBSCAN implementation
├── visualization/          # Plotting utilities
├── config.py              # Configuration parameters
├── main.py                # Main execution script
└── requirements.txt       # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd agnews-umap-hdbscan-clustering
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python main.py
```

This will:
1. Load the AG News dataset
2. Generate embeddings using sentence-transformers
3. Perform dimensionality reduction with UMAP
4. Cluster the data using HDBSCAN
5. Generate visualizations
6. Print cluster summaries

## Configuration

The project can be configured by modifying `config.py`:

- Dataset parameters (name, split, text field)
- Model parameters (embedding model, batch size)
- UMAP parameters (neighbors, components, metric)
- HDBSCAN parameters (min cluster size, min samples)
- Visualization parameters (figure size, colors)

## Output

The script generates:
- Embeddings saved as numpy arrays
- Static plots (PNG)
- Interactive plots (HTML)
- Cluster summaries in the console
- Clustering metrics (number of clusters, noise points, silhouette score)

## Dependencies

- datasets
- sentence-transformers
- umap-learn
- hdbscan
- matplotlib
- seaborn
- scikit-learn
- numpy
- pandas
- plotly
- tqdm

## License

[Your chosen license]

## Contributing

[Your contribution guidelines] 