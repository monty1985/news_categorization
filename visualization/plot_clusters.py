"""
Module for visualizing clustering results in 2D and 3D.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
import logging

from config import CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClusterVisualizer:
    def __init__(self):
        """Initialize the cluster visualizer."""
        os.makedirs(CONFIG['paths']['plots_dir'], exist_ok=True)
        logger.info("Initialized ClusterVisualizer")
    
    def plot_umap_clusters(self, 
                          umap_proj: np.ndarray, 
                          cluster_labels: np.ndarray,
                          texts: List[str] = None,
                          save: bool = True,
                          filename_2d: str = "cluster_plot_2d.png",
                          filename_3d: str = "cluster_plot_3d.html") -> None:
        """
        Create 2D and 3D scatter plots of the UMAP projections with cluster assignments.
        
        Args:
            umap_proj (np.ndarray): UMAP projections (2D or 3D)
            cluster_labels (np.ndarray): Cluster assignments
            texts (List[str], optional): Original texts for hover information
            save (bool): Whether to save the plots
            filename_2d (str): Name of the file to save the 2D plot to
            filename_3d (str): Name of the file to save the 3D plot to
        """
        # Create 2D matplotlib plot if enabled
        if CONFIG['visualization']['plot_2d']:
            plt.figure(figsize=CONFIG['visualization']['figure_size'])
            scatter = sns.scatterplot(
                x=umap_proj[:, 0],
                y=umap_proj[:, 1],
                hue=cluster_labels,
                palette=CONFIG['visualization']['color_palette'],
                s=CONFIG['visualization']['point_size']
            )
            
            plt.title(CONFIG['visualization']['plot_titles']['2d'])
            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")
            plt.legend(title='Cluster', loc='best')
            
            if save:
                save_path = os.path.join(CONFIG['paths']['plots_dir'], filename_2d)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved 2D plot to {save_path}")
            
            plt.show()
        
        # Create interactive 3D plot if enabled and dimensions allow
        if CONFIG['visualization']['plot_3d'] and umap_proj.shape[1] >= 3:
            self.plot_3d_clusters(umap_proj, cluster_labels, texts, save, filename_3d)
    
    def plot_3d_clusters(self,
                        umap_proj: np.ndarray,
                        cluster_labels: np.ndarray,
                        texts: List[str],
                        save: bool = True,
                        filename: str = "cluster_plot_3d.html") -> None:
        """
        Create an interactive 3D visualization of the clusters.
        
        Args:
            umap_proj (np.ndarray): 3D UMAP projections
            cluster_labels (np.ndarray): Cluster assignments
            texts (List[str]): Original texts for hover information
            save (bool): Whether to save the plot
            filename (str): Name of the file to save the plot to
        """
        # Create DataFrame for plotly
        import pandas as pd
        df = pd.DataFrame({
            'UMAP1': umap_proj[:, 0],
            'UMAP2': umap_proj[:, 1],
            'UMAP3': umap_proj[:, 2],
            'Cluster': cluster_labels,
            'Text': texts
        })
        
        # Create interactive 3D plot
        fig = px.scatter_3d(
            df,
            x='UMAP1',
            y='UMAP2',
            z='UMAP3',
            color='Cluster',
            hover_data=['Text'],
            title=CONFIG['visualization']['plot_titles']['3d'],
            size_max=CONFIG['visualization']['marker_size'],
            opacity=CONFIG['visualization']['opacity']
        )
        
        # Update layout for better visualization
        fig.update_layout(
            scene = dict(
                xaxis_title='UMAP 1',
                yaxis_title='UMAP 2',
                zaxis_title='UMAP 3'
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        # Add cluster selection buttons
        if CONFIG['visualization']['interactive']:
            buttons = []
            for cluster in sorted(df['Cluster'].unique()):
                buttons.append(
                    dict(
                        method='update',
                        args=[{'visible': [True if c == cluster else False for c in df['Cluster']]}],
                        label=f'Cluster {cluster}'
                    )
                )
            
            fig.update_layout(
                updatemenus=[
                    dict(
                        type='buttons',
                        showactive=True,
                        buttons=buttons,
                        direction='right',
                        pad={'r': 10, 't': 10},
                        x=0.1,
                        y=1.1
                    )
                ]
            )
        
        if save:
            save_path = os.path.join(CONFIG['paths']['plots_dir'], filename)
            fig.write_html(save_path)
            logger.info(f"Saved 3D plot to {save_path}")
        
        fig.show()

def plot_umap_clusters(umap_proj: np.ndarray, 
                      cluster_labels: np.ndarray,
                      texts: List[str] = None) -> None:
    """
    Convenience function to create cluster visualizations.
    
    Args:
        umap_proj (np.ndarray): UMAP projections (2D or 3D)
        cluster_labels (np.ndarray): Cluster assignments
        texts (List[str], optional): Original texts for hover information
    """
    visualizer = ClusterVisualizer()
    visualizer.plot_umap_clusters(umap_proj, cluster_labels, texts) 