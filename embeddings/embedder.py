"""
Module for generating text embeddings using sentence transformers.
"""

import os
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging
import torch

from config import CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextEmbedder:
    def __init__(self, model_name: str = None):
        """
        Initialize the text embedder with a sentence transformer model.
        
        Args:
            model_name (str): Name of the sentence transformer model to use
        """
        if model_name is None:
            model_name = CONFIG['model']['model_name']
        
        # Check if CUDA is available, otherwise use CPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        self.model = SentenceTransformer(model_name, device=device)
        logger.info(f"Initialized embedder with model: {model_name}")
    
    def get_embeddings(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of text documents to embed
            batch_size (int): Batch size for processing
            
        Returns:
            np.ndarray: Array of embeddings
        """
        if batch_size is None:
            batch_size = CONFIG['model']['parameters']['batch_size']
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        logger.info(f"Using batch size: {batch_size}")
        
        try:
            # Process in smaller chunks with progress bar
            chunk_size = 100  # Process 100 texts at a time
            all_embeddings = []
            
            for i in tqdm(range(0, len(texts), chunk_size), desc="Generating embeddings"):
                chunk = texts[i:i + chunk_size]
                chunk_embeddings = self.model.encode(
                    chunk,
                    batch_size=batch_size,
                    show_progress_bar=False,  # We're using our own progress bar
                    convert_to_numpy=True
                )
                all_embeddings.append(chunk_embeddings)
                logger.info(f"Processed {min(i + chunk_size, len(texts))}/{len(texts)} texts")
            
            embeddings = np.vstack(all_embeddings)
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def save_embeddings(self, embeddings: np.ndarray, filename: str):
        """
        Save embeddings to disk.
        
        Args:
            embeddings (np.ndarray): Embeddings to save
            filename (str): Name of the file to save embeddings to
        """
        os.makedirs(CONFIG['paths']['embeddings_dir'], exist_ok=True)
        save_path = os.path.join(CONFIG['paths']['embeddings_dir'], filename)
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
        load_path = os.path.join(CONFIG['paths']['embeddings_dir'], filename)
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