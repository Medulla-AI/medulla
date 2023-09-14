import numpy as np
from numpy.linalg import norm


def embeddings_compare(source: np.ndarray, 
                       candidates: np.ndarray) -> np.ndarray:
    """
    Calculate cosine similarities between a source vector and multiple candidate vectors.
    
    Args:
        source (np.ndarray): Source vector.
            A 1-dimensional numpy array representing the source vector.
        candidates (np.ndarray): Batch of candidate vectors.
            A 2-dimensional numpy array where each row represents a candidate vector.
        
    Returns:
        np.ndarray: Array of cosine similarity scores.
            A 1-dimensional numpy array containing cosine similarity scores between
            the source vector and each candidate vector in the batch.
    """
    source = np.asarray(source)
    candidates = np.asarray(candidates)
    return np.dot(candidates, source) / (norm(source) * norm(candidates, axis=1))

source = np.array([1.0, 2.0, 3.0])
candidates = np.array([])  # Empty array
print(embeddings_compare(source, candidates))