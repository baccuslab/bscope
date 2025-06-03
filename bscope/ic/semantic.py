import os
from IPython import embed
import json
import tqdm
import numpy as np
import json
import requests
import bscope
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

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

# Helper function to visualize results
def visualize_selection(vector, significant_indices, title):
    plt.figure(figsize=(10, 6))
    plt.plot(vector, 'b-', alpha=0.5)
    plt.plot(significant_indices, vector[significant_indices], 'ro')
    plt.title(title)
    plt.grid(True)
    plt.show()

class SemanticAnalyzer:

    def __init__(self, json_file_path='/mnt/data/ic/node_data_mapping.json'):
        self.data = self.load_data(json_file_path)

    def load_data(self, path):
        """Load synset data from JSON file."""
        with open(path, 'r') as f:
            return json.load(f)

    def indices_helper(self, name, partial_match=False):
        """
        Retrieve synsets by name or partial match, including all indices from descendants.
        """
        # Get all descendants using the get_descendant_nodes function
        all_descendants = self.get_descendant_nodes(name)

        # If no descendants found, try with partial match
        if not all_descendants and partial_match:
            # Try again with partial matching
            direct_matches = {}
            for synset_name, info in self.data.items():
                base_name = synset_name.split('.')[0]
                if name.lower() in base_name.lower():
                    direct_matches[synset_name] = info

            # Return just these direct partial matches if no descendants found
            if direct_matches:
                return direct_matches

        # Process the descendants to add combined indices to each direct match
        results = {}
        direct_matches_found = False

        # First identify direct matches
        for synset_name, info in all_descendants.items():
            base_name = synset_name.split('.')[0]
            if base_name.lower() == name.lower():
                # This is a direct match - store it separately
                results[synset_name] = info.copy(
                )  # Copy to avoid modifying original
                direct_matches_found = True

        # If no direct matches but we have descendants, include all descendants
        if not direct_matches_found and all_descendants:
            # Just return all descendants when no direct matches
            return all_descendants

        # Now, let's add all indices from all descendants to each direct match
        if direct_matches_found:
            # Collect all indices from all descendants
            all_indices = []
            for desc_name, desc_info in all_descendants.items():
                # Get indices from this descendant
                if 'indices' in desc_info and desc_info['indices']:
                    all_indices.extend(desc_info['indices'])

            # Remove duplicates and sort
            all_indices = sorted(list(set(all_indices)))

            # Add these combined indices to each direct match
            for match_name in results:
                results[match_name]['all_descendant_indices'] = all_indices

        # Return the results
        return results

    def get_indices(self, name, partial_match=False):
        """
        Print matching synsets and also show all indices from all descendants.
        Returns a list of all unique indices from all descendants.
        """
        # First get direct matches
        matches = self.indices_helper(name, partial_match)

        if not matches:
            print(f"No synsets found with name '{name}'.")
            return []

        print(f"Found {len(matches)} matching term for '{name}':")

        all_descendants = self.get_descendant_nodes(name)

        # Collect all indices from all descendants
        all_indices = set()
        for desc_name, desc_info in all_descendants.items():
            if 'indices' in desc_info and desc_info['indices']:
                all_indices.update(desc_info['indices'])

        # Sort the indices
        all_indices = sorted(list(all_indices))


        # Return all unique indices for use in mask function
        return all_indices

    def get_parent_nodes(self, name, partial_match=False):
        """Find parent nodes for the given name."""
        matches = self.indices_helper(name, partial_match)
        if not matches:
            print(f"No synsets found with name '{name}'.")
            return {}
        parent_nodes = {}
        for synset_name, info in matches.items():
            full_path = info['path']
            components = full_path.split('.')
            current_path = ""
            for i, component in enumerate(components[:-1]):
                current_path += ("." if i > 0 else "") + component
                for other_name, other_info in self.data.items():
                    if other_info['path'] == current_path:
                        parent_nodes[other_name] = other_info
        return parent_nodes

    def get_descendant_nodes(self, term):
        """
        Find all synsets that are descendants of or match the given term.
        The term is matched against the name part of the synset.
        """
        descendant_nodes = {}
        matched_synsets = []

        # First try to find direct matches
        for synset_name, info in self.data.items():
            base_name = synset_name.split('.')[0]
            if base_name.lower() == term.lower():
                descendant_nodes[synset_name] = info
                matched_synsets.append(synset_name)

        # If we found direct matches, look for their descendants
        for matched_synset in matched_synsets:
            for synset_name, info in self.data.items():
                if synset_name in descendant_nodes:
                    continue  # Skip if already added

                # Check if the synset is a descendant of any matched synset
                if matched_synset in info['path']:
                    descendant_nodes[synset_name] = info

        # If we still didn't find anything or if the term is something like 'aquatic_mammal',
        # try looking for it as a component in paths
        if not descendant_nodes:
            for synset_name, info in self.data.items():
                if term in info['path']:
                    descendant_nodes[synset_name] = info
                    # Now look for descendants of this synset
                    for other_synset, other_info in self.data.items():
                        if synset_name in other_info[
                                'path'] and other_synset != synset_name:
                            descendant_nodes[other_synset] = other_info

        return descendant_nodes


    def get_all_indices_for_search(self, term):
        """
        Get ALL indices from all synsets that have the term in their name or path.
        This includes all direct and indirect descendants.
        """
        all_indices = []

        # Get all descendant nodes
        descendants = self.get_descendant_nodes(term)

        # Extract all indices from these descendants
        for info in descendants.values():
            all_indices.extend(info.get('indices', []))

        return sorted(list(set(all_indices)))

    def get_mask(self, search_term, target_indices):
        """
        Create a boolean mask for target_indices based on whether each index
        is in the descendants of the search_term.
        """
        all_indices = set(self.get_all_indices_for_search(search_term))
        return np.array([idx in all_indices for idx in target_indices])

    def get_all_semantic_masks(self, target_indices):
        """
        Create boolean masks for all nodes using the create_mask method.
        
        Parameters:
        - target_indices: A list of indices to check against each node's descendants
        
        Returns:
        - mask_array: A numpy array of shape (num_nodes, len(target_indices))
                    where mask_array[i, j] is True if target_indices[j] is in node i's descendants
        - node_names: List of node names corresponding to rows in the mask_array
        """
        import numpy as np

        # Get list of all synset names
        node_names = list(self.data.keys())
        num_nodes = len(node_names)

        # Initialize list to store masks
        masks = []

        # For each node, create a mask using the existing create_mask method
        for synset_name in node_names:
            # Get the base name without the synset identifier
            base_name = synset_name.split('.')[0]

            # Use the existing create_mask method
            mask = self.get_mask(base_name, target_indices)
            masks.append(mask)

        # Convert list of masks to numpy array
        mask_array = np.array(masks)

        return mask_array, node_names
    def get_all_imagenet_masks(self, target_indices):
        """
        Create masks for the 1000 ImageNet classes, using the most specific class name for each index.
        
        Parameters:
        - target_indices: A list of indices to check
        
        Returns:
        - mask_array: A numpy array of shape (1000, len(target_indices))
        - imagenet_class_names: List of the 1000 specific class names
        """
        import numpy as np
        
        # For each index, find the most specific class (i.e., the class with the longest path)
        index_to_specific_class = {}
        
        for synset_name, data in self.data.items():
            path_length = len(data['path'].split('.'))
            
            for idx in data['indices']:
                if 0 <= idx < 1000:
                    # If we haven't seen this index before, or if this class is more specific
                    if idx not in index_to_specific_class or \
                    path_length > len(self.data[index_to_specific_class[idx]]['path'].split('.')):
                        index_to_specific_class[idx] = synset_name
        
        # Create a list of class names, one for each index
        class_names = []
        for idx in range(1000):
            if idx in index_to_specific_class:
                class_names.append(index_to_specific_class[idx])
            else:
                print(f"Warning: No class found for index {idx}")
                # Use a placeholder if no class is found
                class_names.append(f"unknown_class_{idx}")
        
        # Create masks for each index
        masks = []
        for idx in range(1000):
            # Simple mask where target index equals current index
            mask = np.array([i == idx for i in target_indices])
            masks.append(mask)
        
        # Convert list of masks to numpy array
        mask_array = np.array(masks)
        
        return mask_array, class_names
    def get_concepts_from_path(self, concept):
        """
        Get all concept names from paths containing the specified concept,
        preserving the original order they appear in each path.
        
        Parameters:
        - concept: String representing the starting concept to search for
        
        Returns:
        - List of concept names in the order they appear in paths
        """
        # Get all descendant nodes for the concept
        descendants = self.get_descendant_nodes(concept)
        
        # Track all unique paths to handle duplicates while preserving order
        all_paths = set()
        for info in descendants.values():
            all_paths.add(info['path'])
        
        # Process each unique path and extract names in order
        ordered_names = []
        seen_names = set()  # To track duplicates
        
        for path in all_paths:
            # Split the path into components
            components = path.split(".")
            
            # Process components in groups of 3 (name, pos, number)
            i = 0
            while i < len(components):
                # Add the name component if not already seen
                if i < len(components) and components[i] not in seen_names:
                    ordered_names.append(components[i])
                    seen_names.add(components[i])
                
                # Skip to the next name
                if i + 2 < len(components) and components[i+1] == 'n':
                    i += 3  # Standard case: skip name, 'n', and number
                else:
                    i += 1  # Fallback: move forward one component
        
        return ordered_names
    
import numpy as np
import matplotlib.pyplot as plt
import bscope
import bscope.ic as bic
from typing import List, Tuple, Dict, Union, Optional


class ModeAnalyzer:
    """
    A class for analyzing contributions using correlation matrices
    and semantic masks.
    """
    
    def __init__(
        self, 
        loadings, 
        dictionary,
        mask_matrix ,
        mask_labels,
        contributions,
        dead_features
    ):
        """
        Initialize the ConceptAnalyzer with model data.
        
        Args:
            loadings: The loadings matrix from SAE (samples × features)
            dictionary: The dictionary/atoms matrix from SAE (features × channels)
            mask_matrix: Semantic mask matrix (concepts × samples)
            mask_labels: Labels for each concept in mask_matrix
            contributions: Contribution matrix (samples × channels)
            dead_features: Optional array indicating dead features to exclude
        """
        self.loadings = loadings
        self.dictionary = dictionary
        self.mask_matrix = mask_matrix  # Ensure mask_matrix is (samples × concepts)
        self.mask_labels = mask_labels
        self.contributions = contributions
        self.dead_features = dead_features
        
        # Calculate correlation matrix between loadings and mask matrix
        self.corr_mtx = np.abs(bscope.mtx_corr(self.loadings, self.mask_matrix))
        self.corr_mtx[np.isnan(self.corr_mtx)] = 0  # Replace NaNs with zeros
    
    def find_concept_indices(self, concept_name: str) -> List[int]:
        """
        Find the indices of concepts that match the given name.
        
        Args:
            concept_name: The name of the concept to find
            
        Returns:
            List of matching concept indices
        """
        matching_indices = []
        for i, label in enumerate(self.mask_labels):
            if label.startswith(concept_name):
                matching_indices.append(i)
        
        return matching_indices
    
    def get_concept_info(self, concept_name: str, select_first: bool = True) -> Tuple[int, str]:
        """
        Get information about a concept.
        
        Args:
            concept_name: The name of the concept to find
            select_first: If True, automatically select the first match if multiple found
            
        Returns:
            Tuple of (concept_index, concept_label)
            
        Raises:
            ValueError: If no matching concepts found or multiple matches found and select_first is False
        """
        matching_indices = self.find_concept_indices(concept_name)
        
        if not matching_indices:
            raise ValueError(f"No concepts found with name '{concept_name}'")
        
        if len(matching_indices) > 1:
            print(f"Found multiple matching concepts:")
            for idx in matching_indices:
                print(f"  {idx}: {self.mask_labels[idx]}")
            
            if select_first:
                concept_idx = matching_indices[0]
                print(f"Using the first match: {self.mask_labels[concept_idx]}")
            else:
                raise ValueError(f"Multiple concepts found with name '{concept_name}'. Set select_first=True to use the first match.")
        else:
            concept_idx = matching_indices[0]
        
        return concept_idx, self.mask_labels[concept_idx]
    
    def get_concept_sample_indices(self, concept_idx: int) -> np.ndarray:
        """
        Get the sample indices for a specific concept.
        
        Args:
            concept_idx: The index of the concept
            
        Returns:
            Array of sample indices where the concept is present
        """
        return np.where(self.mask_matrix[:, concept_idx] == 1)[0]
    
    def get_average_contribution(self, concept_idx: int) -> np.ndarray:
        """
        Get the average contribution for a specific concept.
        
        Args:
            concept_idx: The index of the concept
            
        Returns:
            Average contribution vector across all samples for this concept
        """
        concept_indices = self.get_concept_sample_indices(concept_idx)
        return np.mean(self.contributions[concept_indices], axis=0)

    def get_average_contribution_channels(self, n: int = 10, concept_idx: int = None, method: str = 'argsort') -> np.ndarray:
        """
        Get important channels based on the average contribution of a specific concept.

        Args:
            n: Number of top channels to return (or number of std deviations if method='std')
            concept_idx: The index of the concept
            method: Method to select channels ('argsort' or 'std')
            
        Returns:
            Array of important channel indices
        """
        avg_contribution = self.get_average_contribution(concept_idx)

        if method == 'argsort':
            top_channels = np.argsort(avg_contribution)[-n:][::-1]
        elif method == 'std':
            mean = np.mean(avg_contribution)
            std = np.std(avg_contribution)
            threshold = mean + n * std
            top_channels = np.where(avg_contribution > threshold)[0]
        else:
            raise ValueError(f"Unknown method: {method}. Use 'argsort' or 'std'.")

        return list(top_channels)

    def get_top_modes(self, concept_idx: int, n_modes: int = 5) -> np.ndarray:
        """
        Get the top modes (features) that are most correlated with a concept.
        
        Args:
            concept_idx: The index of the concept
            n_modes: Number of top modes to return
            
        Returns:
            Array of indices of the top n_modes that correlate with the concept
        """
        # Sort modes by correlation with the concept
        concept_modes = np.argsort(self.corr_mtx[:, concept_idx])
        # Get top n_modes (in descending order)
        top_modes = concept_modes[-n_modes:][::-1]
        return top_modes
    def get_channels(self, mode_indices: np.ndarray, n: int = 10, concept_idx=None,
                        method: str = 'argsort') -> List[np.ndarray]:
        """
        Get the important channels for specified modes.
        
        Args:
            mode_indices: Array of mode indices to analyze
            n: Number of top channels per mode to return. If None, returns all channels ranked by importance
            concept_idx: Optional concept index (not used in this implementation)
            method: Method to select channels ('argsort' or 'std')
            
        Returns:
            List of important channel indices across all modes
        """
        important_channels = []
        for mode_idx in mode_indices:
            mode = self.dictionary[mode_idx]
            
            if method == 'argsort':
                if n is None:
                    # Return all channels sorted by importance (highest to lowest)
                    channels = np.argsort(mode)[::-1]  # Full sorted array in descending order
                else:
                    # Get top n channels with highest values
                    channels = np.argsort(mode)[-n:][::-1]
            elif method == 'std':
                if n is None:
                    # For std method with n=None, sort by deviation from mean (highest to lowest)
                    mean = np.mean(mode)
                    deviations = (mode - mean) / np.std(mode)
                    channels = np.argsort(deviations)[::-1]  # Full sorted array in descending order
                else:
                    # Get channels that are x standard deviations above the mean
                    mean = np.mean(mode)
                    std = np.std(mode)
                    threshold = mean + n * std  # using n as the number of std devs
                    channels = np.where(mode > threshold)[0]

            elif method == 'cumsum':
                if n > 1.0:
                    raise ValueError("For 'cumsum' method, n should be a float between 0 and 1.")
                elif n < 0.0:
                    raise ValueError("For 'cumsum' method, n should be a float between 0 and 1.")
                channels = select_significant_indices(mode, method='threshold', param=n)
    
            else:
                raise ValueError(f"Unknown method: {method}. Use 'argsort' or 'std' ")
                
            important_channels.extend(channels)
        important_channels = np.unique(important_channels)
        return list(important_channels)
