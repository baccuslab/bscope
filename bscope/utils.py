import numpy as np


def ei_split(matrix):
    """
    Splits a matrix into positive and negative parts.
    
    Args:
        matrix (np.ndarray): The input matrix to be split.
        
    Returns:
        tuple: Two matrices, one containing the positive values and the other containing the negative values.
    """
    pos = np.where(matrix > 0, matrix, 0)
    neg = np.where(matrix < 0, matrix, 0)

    cat = np.concatenate([neg, pos], axis=1)
    return cat
