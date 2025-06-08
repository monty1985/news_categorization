"""
Module for generating text embeddings using sentence transformers.
"""

import os
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging

from config import MODEL_CONFIG, PATHS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextEmbedder:
    def __init__(self, model_name: str = MODEL_CONFIG['model_name']):
        """
        Initialize the text embedder with a sentence transformer model.
        
        Args:
            model_name (str): Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        logger.info(f"Initialized embedder with model: {model_name}")
    
    def get_embeddings(self, texts: List[str], batch_size: int = MODEL_CONFIG['batch_size']) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of text documents to embed
            batch_size (int): Batch size for processing
            
        Returns:
            np.ndarray: Array of embeddings
        """
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def save_embeddings(self, embeddings: np.ndarray, filename: str):
        """
        Save embeddings to disk.
        
        Args:
            embeddings (np.ndarray): Embeddings to save
            filename (str): Name of the file to save embeddings to
        """
        os.makedirs(PATHS['embeddings_dir'], exist_ok=True)
        save_path = os.path.join(PATHS['embeddings_dir'], filename)
        np.save(save_path, embeddings)
        logger.info(f"Saved embeddings to {save_path}")
    
    def load_embeddings(self, filename: str) -> np.ndarray:
        """
        Load embeddings from disk.
        
        Args:
            filename (str): Name of the file to load embeddings from
            
        Returns:
            np.ndarray: Loaded embeddings
        """
        load_path = os.path.join(PATHS['embeddings_dir'], filename)
        embeddings = np.load(load_path)
        logger.info(f"Loaded embeddings with shape: {embeddings.shape}")
        return embeddings

def get_embeddings(texts: List[str], save: bool = False, filename: str = None) -> np.ndarray:
    """
    Convenience function to generate embeddings for texts.
    
    Args:
        texts (List[str]): List of text documents to embed
        save (bool): Whether to save the embeddings
        filename (str): Name of the file to save embeddings to
        
    Returns:
        np.ndarray: Array of embeddings
    """
    embedder = TextEmbedder()
    embeddings = embedder.get_embeddings(texts)
    
    if save and filename:
        embedder.save_embeddings(embeddings, filename)
    
    return embeddings 