import numpy as np

def mtx_corr(A, B):
    """
    Fast correlation between two matrices.
    Shape[0] must be the same for each
    For example, N x M and N x K matrices
    where N is the number of samples, M is the number of features in A,
    and K is the number of features in B.
    """
    A = A.T
    B = B.T
    # Normalize A and B along the feature axis (columns)
    A_mean = A.mean(axis=1, keepdims=True)
    A_std = A.std(axis=1, keepdims=True)
    A_norm = (A - A_mean) / A_std

    B_mean = B.mean(axis=1, keepdims=True)
    B_std = B.std(axis=1, keepdims=True)
    B_norm = (B - B_mean) / B_std

    correlation_matrix = A_norm @ B_norm.T / A.shape[1]
    return correlation_matrix


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

def compute_participation_ratio(matrix):
    """
    Compute the participation ratio for each row of a matrix.
    
    The participation ratio is defined as:
    PR = (Σᵢ|aᵢ|²)² / (Σᵢ|aᵢ|⁴)
    
    It measures how evenly distributed the elements in each row are.
    When all elements have equal magnitude, PR = n (number of elements).
    When only one element is non-zero, PR = 1.
    
    Parameters:
    -----------
    matrix : numpy.ndarray
        Input matrix with shape (n_rows, n_cols)
    
    Returns:
    --------
    pr : numpy.ndarray
        Participation ratio for each row, shape (n_rows,)
    """
    # Get the shape of the matrix
    n_rows, n_cols = matrix.shape

    # Initialize array to store participation ratios
    participation_ratios = np.zeros(n_rows)

    # Compute participation ratio for each row
    for i in range(n_rows):
        # Get the current row
        row = matrix[i, :]

        # Compute the squared magnitudes
        squared_magnitudes = np.abs(row)**2

        # Compute the sum of squared magnitudes
        sum_squared = np.sum(squared_magnitudes)

        # Compute the sum of fourth powers
        sum_fourth_power = np.sum(squared_magnitudes**2)

        # Compute the participation ratio
        if sum_fourth_power > 0:  # Avoid division by zero
            participation_ratios[i] = sum_squared**2 / sum_fourth_power
        else:
            participation_ratios[i] = 0

    return participation_ratios

def img_norm(image):
    image = image - np.min(image)
    image = image / np.max(image)
    return image

