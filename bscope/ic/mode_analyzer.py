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
