import numpy as np
import torch
Epsilon = 1e-6

def r2_score(x, x_hat, eps=1e-6):
    assert x.shape == x_hat.shape, "Input tensors must have the same shape"
    assert len(x.shape) == 2, "Input tensors must be 2D"

    ss_res = torch.mean((x - x_hat) ** 2)
    ss_tot = torch.mean((x - x.mean()) ** 2)

    r2 = 1 - (ss_res / (ss_tot + eps))

    return torch.mean(r2)

def l2(v, dims=None):
    """
    Compute the L2 norm, across 'dims'.

    Parameters
    ----------
    v : torch.Tensor
        Input tensor.
    dims : tuple, optional
        Dimensions over which to compute the L2 norm, by default None.

    Returns
    -------
    torch.Tensor
        L2 norm of v if dims=None else L2 norm across dims.
    """
    if dims is None:
        return v.square().sum().sqrt()
    return v.square().sum(dims).sqrt()


def l1(v, dims=None):
    """
    Compute the L1 norm, across 'dims'.

    Parameters
    ----------
    v : torch.Tensor
        Input tensor.
    dims : tuple, optional
        Dimensions over which to compute the L1 norm, by default None.

    Returns
    -------
    torch.Tensor
        L1 norm of v if dims=None else L1 for across dims.
    """
    if dims is None:
        return torch.abs(v).sum()
    return torch.abs(v).sum(dims)


def lp(v, p=0.5, dims=None):
    """
    Compute the Lp norm, across 'dims'.

    Parameters
    ----------
    v : torch.Tensor
        Input tensor.
    p : float, optional
        Power of the norm, by default 0.5.
    dims : tuple, optional
        Dimensions over which to compute the Lp norm, by default None.

    Returns
    -------
    torch.Tensor
        Lp norm of v if dims=None else Lp norm across dims.
    """
    if dims is None:
        return torch.norm(v, p)
    return torch.norm(v, p, dims)


def avg_l2_loss(x, x_hat):
    """
    Compute the L2 loss, averaged across samples.

    Parameters
    ----------
    x : torch.Tensor
        Original input tensor of shape (batch_size, d).
    x_hat : torch.Tensor
        Reconstructed input tensor of shape (batch_size, d).

    Returns
    -------
    float
        Average L2 loss per sample.
    """
    assert x.shape == x_hat.shape, "Input tensors must have the same shape"
    assert len(x.shape) == 2, "Input tensors must be 2D"
    return torch.mean(l2(x - x_hat, 1)).item()


def avg_l1_loss(x, x_hat):
    """
    Compute the L1 loss, averaged across samples.

    Parameters
    ----------
    x : torch.Tensor
        Original input tensor of shape (batch_size, d).
    x_hat : torch.Tensor
        Reconstructed input tensor of shape (batch_size, d).

    Returns
    -------
    float
        Average L1 loss per sample.
    """
    assert x.shape == x_hat.shape, "Input tensors must have the same shape"
    assert len(x.shape) == 2, "Input tensors must be 2D"
    return torch.mean(l1(x - x_hat, 1)).item()


def relative_avg_l2_loss(x, x_hat, epsilon=Epsilon):
    """
    Compute the relative reconstruction loss, average across samples.

    The first argument is considered as the true value. The order of the arguments
    is important as the loss is asymmetric:

    ||x - y||_2 / (||x||_2 + epsilon).

    Parameters
    ----------
    x : torch.Tensor
        Original input tensor of shape (batch_size, d).
    x_hat : torch.Tensor
        Reconstructed input tensor of shape (batch_size, d).
    epsilon : float, optional
        Small value to avoid division by zero, by default 1e-6.

    Returns
    -------
    float
        Average relative L2 loss per sample.
    """
    assert x.shape == x_hat.shape, "Input tensors must have the same shape"
    assert len(x.shape) == 2, "Input tensors must be 2D"

    l2_err_per_sample = l2(x - x_hat, 1)
    l2_per_sample = l2(x, 1)

    return torch.mean(l2_err_per_sample / (l2_per_sample + epsilon)).item()


def relative_avg_l1_loss(x, x_hat, epsilon=Epsilon):
    """
    Compute the relative reconstruction loss, average across samples.

    The first argument is considered as the true value. The order of the arguments
    is important as the loss is asymmetric:

    ||x - y||_1 / (||x||_1 + epsilon).

    Parameters
    ----------
    x : torch.Tensor
        Original input tensor of shape (batch_size, d).
    x_hat : torch.Tensor
        Reconstructed input tensor of shape (batch_size, d).
    epsilon : float, optional
        Small value to avoid division by zero, by default 1e-6.

    Returns
    -------
    float
        Average relative L1 loss per sample.
    """
    assert x.shape == x_hat.shape, "Input tensors must have the same shape"
    assert len(x.shape) == 2, "Input tensors must be 2D"

    l1_err_per_sample = l1(x - x_hat, 1)
    l1_per_sample = l1(x, 1)

    return torch.mean(l1_err_per_sample / (l1_per_sample + epsilon)).item()


def l0(x, dims=None):
    """
    Compute the average number of non-zero elements.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    dims : tuple, optional
        Dimensions over which to average the l0 norm, by default average across all dims.

    Returns
    -------
    torch.Tensor
        Average l0 norm if dims=None else l0 across dims.
    """
    if dims is None:
        return torch.mean((x != 0).float())
    return torch.mean((x != 0).float(), dims)


def l0_eps(x, dims=None, threshold=1e-6):
    """
    Compute the l0 norm allowing for an epsilon tolerance.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    dims : tuple, optional
        Dimensions over which to average the l0 norm, by default all dims.
    threshold : float, optional
        Epsilon tolerance, by default 1e-6.

    Returns
    -------
    torch.Tensor
        Average l0 if dims=None else l0 across dims.
    """
    if dims is None:
        return torch.mean((torch.abs(x) >= threshold).float())
    return torch.mean((torch.abs(x) >= threshold).float(), dims)


def l1_l2_ratio(x, dims=-1):
    """
    Compute the L1/L2 ratio of a tensor. By default, the ratio is computed across
    the last dimension. This score is a useful metric to evaluate the sparsity of
    a tensor. It is however sensitive to the dimensions of the data, for an unbiased
    metric, consider using the Hoyer score.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    dims : tuple, optional
        Dimensions over which to compute the ratio, by default -1.

    Returns
    -------
    torch.Tensor
        the l1/l2 ratio.
    """
    l1_norm = l1(x, dims)
    l2_norm = l2(x, dims) + Epsilon

    return l1_norm / l2_norm


def _max_non_diagonal(matrix):
    """
    Compute the maximum value of non-diagonal elements in a square matrix.

    Parameters
    ----------
    matrix : torch.Tensor
        Input square matrix of shape (n, n).

    Returns
    -------
    float
        Maximum non-diagonal element.
    """
    assert matrix.shape[0] == matrix.shape[1], "Input must be a square matrix"

    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
    non_diagonal_values = matrix[mask]

    return torch.max(non_diagonal_values).item()
def hoyer(x):
    """
    Compute the Hoyer sparsity of a tensor. The hoyer score include the dimension normalization
    factor. A score of 1 indicates a perfectly sparse representation, while a score of 0 indicates
    a dense representation.

    hoyer(x) = d^0.5 - (||x||_1 / ||x||_2) / (d^0.5 - 1).

    The score is computed across the last dimension.

    Parameters
    ----------
    x : torch.Tensor
        A 2D tensor of shape (batch_size, d).

    Returns
    -------
    torch.Tensor (batch_size,)
        Hoyer sparsity for each vector in the batch.
    """
    assert len(x.shape) == 2, "Input tensor must be 2D"

    d_sqrt = torch.sqrt(torch.tensor(x.shape[1]))
    l1_l2 = l1_l2_ratio(x, 1)

    score = (d_sqrt - l1_l2) / (d_sqrt - 1)

    return score

def dictionary_collinearity(dictionary):
    """
    Compute the collinearity of a dictionary.

    Parameters
    ----------
    dictionary : torch.Tensor
        Dictionary tensor of shape (num_codes, dim).

    Returns
    -------
    max_collinearity : float
        Maximum collinearity across dictionary elements (non diagonal).
    cosine_similarity_matrix : torch.Tensor
        Matrix of cosine similarities across dictionary elements.
    """
    assert len(dictionary.shape) == 2, "Input tensor must be 2D"

    normalized_dict = dictionary / (dictionary.norm(dim=1, keepdim=True) + Epsilon)

    cosine_similarity_matrix = torch.matmul(normalized_dict, normalized_dict.T)
    max_collinearity = _max_non_diagonal(torch.abs(cosine_similarity_matrix))

    return max_collinearity, cosine_similarity_matrix.detach()


def wasserstein_1d(x1, x2):
    """
    Compute the 1D Wasserstein-1 distance between two sets of codes and average
    across dimensions.

    Parameters
    ----------
    x1 : torch.Tensor
        First set of samples of shape (num_samples, d).
    x2 : torch.Tensor
        Second set of samples of shape (num_samples, d).

    Returns
    -------
    torch.Tensor
        Wasserstein distance.
    """
    assert x1.shape == x2.shape, "The two sets must have the same shape"
    assert len(x1.shape) == 2, "Input tensors must be 2D"

    x1_sorted, _ = torch.sort(x1, dim=0)
    x2_sorted, _ = torch.sort(x2, dim=0)

    # avg of wasserstein across dimensions
    dist = torch.mean(torch.abs(x1_sorted - x2_sorted))

    return dist


def cosine_distance_matrix(x, y):
    """
    Compute the cosine distance matrix between two sets of vectors.

    Parameters
    ----------
    x : torch.Tensor
        First set of vectors of shape (num_vectors_x, dim).
    y : torch.Tensor
        Second set of vectors of shape (num_vectors_y, dim).

    Returns
    -------
    torch.Tensor
        Cosine distance matrix of shape (num_vectors_x, num_vectors_y).
    """
    assert x.shape[1] == y.shape[1], "Input vectors must have the same dimensionality"
    assert len(x.shape) == 2 and len(y.shape) == 2, "Input tensors must be 2D"

    x_normalized = x / x.norm(dim=1, keepdim=True)
    y_normalized = y / y.norm(dim=1, keepdim=True)

    cosine_similarity = torch.matmul(x_normalized, y_normalized.T)
    cosine_distance = 1 - cosine_similarity

    return cosine_distance


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


