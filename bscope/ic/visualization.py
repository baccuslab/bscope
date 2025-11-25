import numpy as np
from scipy.ndimage import median_filter, gaussian_filter
from IPython import embed

def normalize(array):
    """
    Normalize a numpy array to the range [0, 1].

    Parameters:
    - array: np.ndarray
        The input array to be normalized.

    Returns:
    - normalized_array: np.ndarray
        The normalized array.
    """
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = (array - min_val) / (max_val - min_val + 1e-8)
    return normalized_array

def generate_cwirf_mask(stim, mode_map, contrast, filter_size, grayscale=False):

    mode_map = np.abs(mode_map)
    mode_map /= (np.max(mode_map) + 1e-10)

    if grayscale:
        mode_map = np.mean(mode_map, axis=2)
        mode_map = median_filter(mode_map[:, :], size=filter_size)[:, :, np.newaxis]
        mode_map -= np.min(mode_map)
        mode_map /= (np.max(mode_map) + 1e-10)
        mode_map *= contrast
        cwirf_mask = normalize(stim) * mode_map

    else:
        mode_map = np.array([median_filter(mode_map[:, :, c], size=filter_size) for c in range(3)]).transpose(1,2,0)
        mode_map -= np.min(mode_map)
        mode_map /= (np.max(mode_map) + 1e-10)
        mode_map *= contrast 

        cwirf_mask = normalize(stim) * mode_map
    return cwirf_mask

def generate_mode_map(contributions, irfs, direction_only=True, contribution_sign_policy='preserve', color_policy='preserve', irf_sign_policy='preserve', map_sign_policy='preserve'):
    # Normalize contributions so that the highest contribution is 1
    contributions /= (np.max(np.abs(contributions)) + 1e-10)

    # Get shapes
    n_chan, h, w = contributions.shape

    if contribution_sign_policy == 'absolute':
        contributions = np.abs(contributions)
    elif contribution_sign_policy == 'positive':
        contributions[contributions < 0] = 0
    elif contribution_sign_policy == 'negative':
        contributions[contributions > 0] = 0
        contributions = np.abs(contributions)
    elif contribution_sign_policy == 'preserve':
        pass
    
    # Normalize contributions for each unit
    for chan in range(n_chan):
        for i in range(h):
            for j in range(w):
                irf = irfs[chan, i, j, :, :, :]
                # normalize the irf by its norm

                if direction_only:
                    irf_norm = np.linalg.norm(irf)
                    irf = irf / (irf_norm + 1e-10)

                if irf_sign_policy == 'absolute':
                    irf = np.abs(irf)
                elif irf_sign_policy == 'positive':
                    irf[irf < 0] = 0
                elif irf_sign_policy == 'preserve':
                    pass
                irfs[chan, i, j, :, :, :] = irf

        print(' + normalized irfs for direction only mode visualization')
    
    contribution_weighted_irfs = np.zeros_like(irfs)
    for chan in range(n_chan):
        for i in range(h):
            for j in range(w):
                irf = irfs[chan, i, j, :, :, :]
                contrib = contributions[chan, i, j]
                contribution_weighted_irfs[chan, i, j, :, :, :] = irf * contrib

    # Aggregate contribution-weighted irfs across all channels and spatial locations

    importance_map = np.sum(contribution_weighted_irfs, axis=(0, 1, 2))

    if map_sign_policy == 'absolute':
        importance_map = np.abs(importance_map)
    elif map_sign_policy == 'positive':
        importance_map[importance_map < 0] = 0
    elif map_sign_policy == 'negative':
        importance_map[importance_map > 0] = 0
        importance_map = np.abs(importance_map)
    elif map_sign_policy == 'preserve':
        pass
    
    importance_map = importance_map.transpose(1, 2, 0)
    if color_policy == 'preserve':
        pass
    elif color_policy == 'max':
        importance_map = np.max(importance_map, axis=2)
        importance_map = importance_map[:, :, np.newaxis]
    elif color_policy == 'mean':
        importance_map = np.mean(importance_map, axis=2)
        importance_map = importance_map[:, :, np.newaxis]

    return importance_map




                

    embed()
    # return importance_map.astype(np.uint8)

def normalize_symmetric(array):
    array /=  np.max(np.abs(array))
    array *= 0.5
    array += 0.5
    return array




def find_high_class_loadings(loadings, count_threshold=5, loading_threshold=0.5):
    block_size = 50
    n_blocks = len(loadings) // block_size
    img_indices = []
    block_indices = []
    for i in range(n_blocks):
        block_loadings = loadings[i*block_size:(i+1)*block_size]
        count = np.sum(block_loadings > loading_threshold)
        if count >= count_threshold:
            block_indices.append(i)
            for j in range(block_size):
                if block_loadings[j] > loading_threshold:
                    img_indices.append(i*block_size + j)


    return img_indices, block_indices

