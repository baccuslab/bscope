import numpy as np
from scipy import integrate, signal
import torch
import matplotlib as mpl
import re
Epsilon = 1e-6


def sort_data(x, y):
    sorted_indices = np.argsort(x)
    return np.array(x)[sorted_indices], np.array(y)[sorted_indices]
def compute_auc(percentages, accuracies, method='trapz'):
    x,y = sort_data(percentages, accuracies)
    
    if method == 'trapz':
        # Trapezoidal rule - most common and robust
        auc = np.trapz(y, x)
        
    elif method == 'simps':
        # Simpson's rule - more accurate for smooth curves
        auc = integrate.simps(y, x)
        
    return auc
def select_significant_indices(vector, method='threshold', param=0.8, min_indices=1, max_indices=None):
    """
    Select indices that contribute most to the overall sum of the vector.
    
    Parameters:
    -----------
    vector : array-like
        Input vector of values
    method : str, optional (default='threshold')
        Method to use for selecting indices:
        - 'threshold': Select indices that contribute to param (e.g. 0.8) of the total sum
        - 'percentile': Select indices above the param (e.g. 90th) percentile
        - 'top_n': Select the top param (e.g. 10) indices by value
        - 'kmeans': Use k-means clustering to separate significant from non-significant values
        - 'otsu': Use Otsu's thresholding method (common in image processing)
    param : float or int, optional
        Parameter specific to the chosen method
    min_indices : int, optional (default=1)
        Minimum number of indices to return
    max_indices : int, optional (default=None)
        Maximum number of indices to return
        
    Returns:
    --------
    significant_indices : numpy.ndarray
        Array of indices that contribute most to the total sum
    """
    vector = np.asarray(vector)
    n = len(vector)
    
    if max_indices is None:
        max_indices = n
    
    # Handle edge cases
    if n == 0:
        return np.array([], dtype=int)
    
    if method == 'threshold':
        # Sort indices by their values in descending order
        sorted_indices = np.argsort(-vector)
        cumsum = np.cumsum(vector[sorted_indices])
        total_sum = cumsum[-1]
        
        # Find how many indices we need to reach the threshold
        if total_sum == 0:  # Handle zero-sum case
            return np.array([0], dtype=int)
        
        # Find indices that contribute to param (e.g. 80%) of the total sum
        threshold_idx = np.searchsorted(cumsum / total_sum, param)
        threshold_idx = max(min_indices, min(threshold_idx + 1, max_indices))
        return sorted_indices[:threshold_idx]
    
    elif method == 'percentile':
        # Select indices above a certain percentile
        threshold = np.percentile(vector, 100 - param)
        significant_indices = np.where(vector >= threshold)[0]
        
        # Adjust if we have too few or too many indices
        if len(significant_indices) < min_indices:
            sorted_indices = np.argsort(-vector)
            significant_indices = sorted_indices[:min_indices]
        elif len(significant_indices) > max_indices:
            sorted_significant = sorted(significant_indices, key=lambda i: -vector[i])
            significant_indices = np.array(sorted_significant[:max_indices])
            
        return significant_indices
    
    elif method == 'top_n':
        # Select the top N indices
        n_indices = min(max(min_indices, int(param)), max_indices)
        return np.argsort(-vector)[:n_indices]
    
    elif method == 'kmeans':
        # Use k-means to separate significant from non-significant values
        from sklearn.cluster import KMeans
        
        # Reshape for k-means
        X = vector.reshape(-1, 1)
        
        # Try to estimate the optimal number of clusters if not specified
        if param == 0:
            from sklearn.metrics import silhouette_score
            scores = []
            for k in range(2, min(10, n)):
                kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
                score = silhouette_score(X, kmeans.labels_)
                scores.append(score)
            param = np.argmax(scores) + 2  # Add 2 because we started from k=2
        
        # Apply k-means clustering
        kmeans = KMeans(n_clusters=int(param), random_state=42).fit(X)
        
        # Get the cluster with the highest mean value
        cluster_means = [np.mean(vector[kmeans.labels_ == i]) for i in range(int(param))]
        top_cluster = np.argmax(cluster_means)
        
        # Get indices belonging to the top cluster
        significant_indices = np.where(kmeans.labels_ == top_cluster)[0]
        
        # Sort by value within the cluster and apply min/max constraints
        significant_indices = sorted(significant_indices, key=lambda i: -vector[i])
        significant_indices = np.array(significant_indices[:max_indices])
        
        if len(significant_indices) < min_indices:
            sorted_indices = np.argsort(-vector)
            missing = min_indices - len(significant_indices)
            extra_indices = [i for i in sorted_indices if i not in significant_indices][:missing]
            significant_indices = np.append(significant_indices, extra_indices)
            
        return significant_indices
    
    elif method == 'otsu':
        # Otsu's method to find optimal threshold
        # Normalize to 0-255 range for Otsu's algorithm
        if np.max(vector) == np.min(vector):
            # Handle constant vectors
            return np.array([0], dtype=int)
            
        normalized = ((vector - np.min(vector)) / (np.max(vector) - np.min(vector)) * 255).astype(np.uint8)
        threshold = signal.threshold_otsu(normalized)
        
        # Convert back to original scale
        original_threshold = threshold / 255 * (np.max(vector) - np.min(vector)) + np.min(vector)
        significant_indices = np.where(vector >= original_threshold)[0]
        
        # Apply min/max constraints
        if len(significant_indices) < min_indices:
            sorted_indices = np.argsort(-vector)
            significant_indices = sorted_indices[:min_indices]
        elif len(significant_indices) > max_indices:
            sorted_significant = sorted(significant_indices, key=lambda i: -vector[i])
            significant_indices = np.array(sorted_significant[:max_indices])
            
        return significant_indices
    elif method == 'std':
    # Select indices above param standard deviations from the mean
        mean_val = np.mean(vector)
        std_val = np.std(vector)
        
        if std_val == 0:  # Handle constant vectors
            sorted_indices = np.argsort(-vector)
            return sorted_indices[:min_indices]
        
        threshold = mean_val + param * std_val
        significant_indices = np.where(vector >= threshold)[0]
        
        # Adjust if we have too few or too many indices
        if len(significant_indices) < min_indices:
            sorted_indices = np.argsort(-vector)
            significant_indices = sorted_indices[:min_indices]
        elif len(significant_indices) > max_indices:
            sorted_significant = sorted(significant_indices, key=lambda i: -vector[i])
            significant_indices = np.array(sorted_significant[:max_indices])
            
        return significant_indices
    
    else:
        raise ValueError(f"Unknown method: {method}")



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


def ei_split(matrix, dim=1):
    """
    Splits a matrix into positive and negative parts.
    
    Args:
        matrix (np.ndarray): The input matrix to be split.
        
    Returns:
        tuple: Two matrices, one containing the positive values and the other containing the negative values.
    """
    pos = np.where(matrix > 0, matrix, 0)
    neg = np.where(matrix < 0, matrix, 0)
    
    cat = np.concatenate([neg, pos], axis=dim)
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
import numpy as np
from scipy import stats


def normalized_mean_square_error(original, reconstruction):
    """
    Compute the Normalized Mean Square Error between original signal and reconstruction.
    
    Parameters:
    -----------
    original : numpy.ndarray
        Original signal of shape (num_samples, feature_dim)
    reconstruction : numpy.ndarray
        Reconstructed signal of shape (num_samples, feature_dim)
        
    Returns:
    --------
    float
        The normalized mean square error
    """
    if original.shape != reconstruction.shape:
        raise ValueError("Original and reconstruction must have the same shape")
    
    # Calculate mean square error
    mse = np.mean(np.square(original - reconstruction))
    
    # Normalize by the power of the original signal
    original_power = np.mean(np.square(original))
    
    # Avoid division by zero
    if original_power == 0:
        return float('inf')
    
    nmse = mse / original_power
    
    return nmse


def cross_entropy_degradation(original, reconstruction, epsilon=1e-10):
    """
    Compute the Cross-Entropy Degradation between original signal and reconstruction.
    This function assumes the signals represent probability distributions or can be 
    normalized to become distributions.
    
    Parameters:
    -----------
    original : numpy.ndarray
        Original signal of shape (num_samples, feature_dim)
    reconstruction : numpy.ndarray
        Reconstructed signal of shape (num_samples, feature_dim)
    epsilon : float
        Small constant to avoid numerical issues with log(0)
        
    Returns:
    --------
    float
        The cross-entropy degradation
    """
    if original.shape != reconstruction.shape:
        raise ValueError("Original and reconstruction must have the same shape")
    
    # Ensure the signals are treated as probability distributions
    # by normalizing along the feature dimension
    orig_norm = original / (np.sum(original, axis=1, keepdims=True) + epsilon)
    recon_norm = reconstruction / (np.sum(reconstruction, axis=1, keepdims=True) + epsilon)
    
    # Calculate cross-entropy
    cross_entropy = -np.sum(orig_norm * np.log(recon_norm + epsilon)) / original.shape[0]
    
    # Calculate entropy of original signal
    entropy_orig = -np.sum(orig_norm * np.log(orig_norm + epsilon)) / original.shape[0]
    
    # Cross-entropy degradation: difference between cross-entropy and entropy
    # Higher values indicate worse reconstruction
    degradation = cross_entropy - entropy_orig
    
    return degradation


def style_plot(ax):
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Ensure linewidth is applied to remaining spines
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    # Adjust tick parameters
    ax.tick_params(width=2, size=6)# Configure global matplotlib parameters
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['lines.markersize'] = 12
    mpl.rcParams['font.size'] = 18
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16


def parse_config(config_name):
    """Parse hyperparams from config directory name."""
    params = {
        'config_name': config_name,
        'threshold': None,
        'N': None,
        'atom_l1': None,
        'mlp_size': None,
        'nonneg': None,
        'sweep_type': None
    }
    
    # Standard format
    standard_match = re.match(r'hypersweep_(\d+\.?\d*)_(\d+)_(\d+\.?\d*e?-?\d*)', config_name)
    if standard_match:
        params['threshold'] = float(standard_match.group(1))
        params['N'] = int(standard_match.group(2))
        params['atom_l1'] = float(standard_match.group(3)) if standard_match.group(3) != '0' else 0.0
        params['sweep_type'] = 'grid'
        return params
    
    # MLP format
    mlp_match = re.match(r'hypersweep_mlpsize_(\d+)_nonneg_(True|False)_(\d+\.?\d*)_(\d+)_(\d+\.?\d*e?-?\d*)', config_name)
    if mlp_match:
        params['mlp_size'] = int(mlp_match.group(1))
        params['nonneg'] = mlp_match.group(2) == 'True'
        params['threshold'] = float(mlp_match.group(3))
        params['N'] = int(mlp_match.group(4))
        params['atom_l1'] = float(mlp_match.group(5)) if mlp_match.group(5) != '0' else 0.0
        params['sweep_type'] = 'mlp'
        return params
    
    # No match - return dict with all fields as None (except config_name)
    return params
