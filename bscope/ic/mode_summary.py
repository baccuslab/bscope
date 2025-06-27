import numpy as np
from IPython import embed
import h5py as h5
import matplotlib.pyplot as plt
import bscope
import bscope.ic as bic
from typing import List, Tuple, Dict, Union, Optional
from dataclasses import dataclass

@dataclass
class LayerSummary:
    """
    A class to summarize the information of a single layer in the mode summary.
    
    Attributes:
        corr_mtx: Correlation matrix between loadings and mask matrix
        r2: R-squared values for the layer
        loadings: Loadings matrix from SAE (samples × features)
        dictionary: Dictionary/atoms matrix from SAE (features × channels)
    """
    corr_mtx: np.ndarray
    loadings: np.ndarray
    dictionary: np.ndarray
    idx: Optional[int] = None  # Optional index for the layer, if needed
    
    def __post_init__(self):
        self.corr_mtx[np.isnan(self.corr_mtx)] = 0  # Replace NaNs with zeros

class ModeSummary:
    def __init__(self, h5_path):
        """
        Initialize the ModeSummary with a path to an HDF5 file.
        
        Args:
            h5_path: Path to the HDF5 file containing mode summary data
        """
        self.h5_path = h5_path
        self.file = h5.File(h5_path, 'r')
        
        # Load mask labels if available
        if 'mask_labels' in self.file:
            raw_labels = self.file['mask_labels'][:]
            # Convert bytes to strings, strip whitespace, and extract base name
            self.mask_labels = [label.decode('utf-8').strip().split('.')[0] if isinstance(label, bytes) 
                            else str(label).strip().split('.')[0] for label in raw_labels]
        else:
            self.mask_labels = None
        self.mask_matrix = self.file['mask_matrix'][:] if 'mask_matrix' in self.file else None
        
        self.layer_idxs = np.sort([int(l) for l in list(self.file['layers'].keys())])
        self.layers = []
        for layer_idx in self.layer_idxs:
            layer_key = str(layer_idx)
            layer_data = self.file['layers'][layer_key]
            corr_mtx = layer_data['corr_mtx'][:]
            loadings = layer_data['loadings'][:]
            dictionary = layer_data['dictionary'][:]
            
            self.layers.append(LayerSummary(corr_mtx, loadings, dictionary, layer_idx))



class ModeAnalyzer:
    """
    A class for analyzing modes and channels using ModeSummary data.
    """
    

    def __init__(self, mode_summary: ModeSummary, contributions=None):
            """
            Initialize the ModeAnalyzer with a ModeSummary instance.
            
            Args:
                mode_summary: ModeSummary instance containing the data
                contributions: Optional contributions array for sample analysis
            """
            self.mode_summary = mode_summary
            self.contributions = contributions
    
    def find_concept_indices(self, concept_name: str) -> List[int]:
        """
        Find the indices of concepts that match the given name.
        
        Args:
            concept_name: The name of the concept to find
            
        Returns:
            List of matching concept indices
        """
        if self.mode_summary.mask_labels is None:
            raise ValueError("No mask labels available in the ModeSummary")
            
        matching_indices = []
        concept_name_lower = concept_name.lower()
        
        for i, label in enumerate(self.mode_summary.mask_labels):
            # Handle both string and bytes labels
            if isinstance(label, bytes):
                label_str = label.decode('utf-8')
            else:
                label_str = str(label)
            
            # Extract base name (same as SemanticAnalyzer)
            base_name = label_str.split('.')[0].strip().lower()
            
            if base_name == concept_name_lower:
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
                print(f"  {idx}: {self.mode_summary.mask_labels[idx]}")
            
            if select_first:
                concept_idx = matching_indices[0]
                print(f"Using the first match: {self.mode_summary.mask_labels[concept_idx]}")
            else:
                raise ValueError(f"Multiple concepts found with name '{concept_name}'. Set select_first=True to use the first match.")
        else:
            concept_idx = matching_indices[0]
        
        return concept_idx, self.mode_summary.mask_labels[concept_idx]
    
    def get_layer(self, layer_idx: int) -> LayerSummary:
        """
        Get a specific layer from the ModeSummary.
        
        Args:
            layer_idx: Index of the layer to retrieve
            
        Returns:
            LayerSummary for the specified layer
            
        Raises:
            ValueError: If layer_idx is not found
        """
        for layer in self.mode_summary.layers:
            if layer.idx == layer_idx:
                return layer
        
        raise ValueError(f"Layer {layer_idx} not found. Available layers: {self.mode_summary.layer_idxs}")
    
    def get_top_modes(self, layer_idx: int, concept_name: str, method: str = 'percentile', 
                     param: float = 0.7, min_indices: int = 1, max_indices: int = 50,
                     select_first: bool = True) -> np.ndarray:
        """
        Get top modes for a specific layer and concept using select_significant_indices.
        
        Args:
            layer_idx: Index of the layer
            concept_name: Name of the concept to analyze
            method: Method for select_significant_indices ('threshold', 'percentile', etc.)
            param: Parameter for the selection method
            min_indices: Minimum number of indices to return
            max_indices: Maximum number of indices to return
            select_first: Whether to auto-select first concept match
            
        Returns:
            Array of mode indices
        """
        # Get the layer
        layer = self.get_layer(layer_idx)
        
        # Get concept info
        concept_idx, concept_label = self.get_concept_info(concept_name, select_first)
        
        # Get correlations for this concept (equivalent to your rs array)
        correlations = layer.corr_mtx[:, concept_idx]
        
        # Use select_significant_indices to get top modes
        modes = bscope.select_significant_indices(
            correlations, 
            method=method, 
            param=param, 
            min_indices=min_indices, 
            max_indices=max_indices
        )
        
        return modes
    
    def get_top_channels(self, layer_idx: int, concept_name: str, 
                        mode_method: str = 'percentile', mode_param: float = 0.7,
                        mode_min_indices: int = 1, mode_max_indices: int = 50,
                        channel_method: str = 'percentile', channel_param: float = 0.5,
                        channel_min_indices: int = 1, channel_max_indices: int = 50,
                        select_first: bool = True) -> List[int]:
        """
        Get top channels for a specific layer and concept by first getting top modes,
        then getting channels from those modes' dictionary vectors.
        
        Args:
            layer_idx: Index of the layer
            concept_name: Name of the concept to analyze
            mode_method: Method for selecting modes
            mode_param: Parameter for mode selection
            mode_min_indices: Minimum number of modes
            mode_max_indices: Maximum number of modes
            channel_method: Method for selecting channels from each mode
            channel_param: Parameter for channel selection
            channel_min_indices: Minimum number of channels per mode
            channel_max_indices: Maximum number of channels per mode
            select_first: Whether to auto-select first concept match
            
        Returns:
            List of unique channel indices across all selected modes
        """
        # Get the layer
        layer = self.get_layer(layer_idx)
        
        # Get top modes first
        top_modes = self.get_top_modes(
            layer_idx, concept_name, mode_method, mode_param,
            mode_min_indices, mode_max_indices, select_first
        )
        
        # Collect channels from all modes
        all_channels = []
        
        for mode_idx in top_modes:
            # Get the dictionary vector (atom) for this mode
            atom = layer.dictionary[mode_idx, :]
            
            # Get top channels for this atom
            top_channels = bscope.select_significant_indices(
                atom,
                method=channel_method,
                param=channel_param,
                min_indices=channel_min_indices,
                max_indices=channel_max_indices
            )
            
            all_channels.extend(top_channels)
        
        # Return unique channels
        return list(np.unique(all_channels))
    
    def get_mode_correlations(self, layer_idx: int, concept_name: str, 
                             select_first: bool = True) -> np.ndarray:
        """
        Get correlation values for all modes with a specific concept.
        
        Args:
            layer_idx: Index of the layer
            concept_name: Name of the concept
            select_first: Whether to auto-select first concept match
            
        Returns:
            Array of correlation values (equivalent to your rs array)
        """
        layer = self.get_layer(layer_idx)
        concept_idx, _ = self.get_concept_info(concept_name, select_first)
        return layer.corr_mtx[:, concept_idx]
    
    def get_concept_sample_indices(self, concept_idx: int) -> np.ndarray:
        """
        Get the sample indices for a specific concept.
        
        Args:
            concept_idx: The index of the concept
            
        Returns:
            Array of sample indices where the concept is present
        """
        return np.where(self.mask_matrix[:, concept_idx] == 1)[0]
    
    def get_average_contribution(self, concept_name: str, select_first: bool = True) -> np.ndarray:
        """
        Get the average contribution for a specific concept.
        
        Args:
            concept_name: The name of the concept
            select_first: Whether to auto-select first concept match
            
        Returns:
            Average contribution vector across all samples for this concept
        """
        # Get concept index from name
        concept_idx, _ = self.get_concept_info(concept_name, select_first)
        
        # Get sample indices for this concept
        concept_indices = self.get_concept_sample_indices(concept_idx)
        
        # Return average contribution across samples
        return np.mean(self.contributions[concept_indices], axis=0)

# class ModeAnalyzer:
#     """
#     A class for analyzing contributions using correlation matrices
#     and semantic masks.
#     """
    
#     def __init__(
#         self, 
#         loadings, 
#         dictionary,
#         mask_matrix ,
#         mask_labels,
#         contributions,
#         dead_features
#     ):
#         """
#         Initialize the ConceptAnalyzer with model data.
        
#         Args:
#             loadings: The loadings matrix from SAE (samples × features)
#             dictionary: The dictionary/atoms matrix from SAE (features × channels)
#             mask_matrix: Semantic mask matrix (concepts × samples)
#             mask_labels: Labels for each concept in mask_matrix
#             contributions: Contribution matrix (samples × channels)
#             dead_features: Optional array indicating dead features to exclude
#         """
#         self.loadings = loadings
#         self.dictionary = dictionary
#         self.mask_matrix = mask_matrix  # Ensure mask_matrix is (samples × concepts)
#         self.mask_labels = mask_labels
#         self.contributions = contributions
#         self.dead_features = dead_features
        
#         # Calculate correlation matrix between loadings and mask matrix
#         self.corr_mtx = np.abs(bscope.mtx_corr(self.loadings, self.mask_matrix))
#         self.corr_mtx[np.isnan(self.corr_mtx)] = 0  # Replace NaNs with zeros
    
#     def find_concept_indices(self, concept_name: str) -> List[int]:
#         """
#         Find the indices of concepts that match the given name.
        
#         Args:
#             concept_name: The name of the concept to find
            
#         Returns:
#             List of matching concept indices
#         """
#         matching_indices = []
#         for i, label in enumerate(self.mask_labels):
#             if label.startswith(concept_name):
#                 matching_indices.append(i)
        
#         return matching_indices
    
#     def get_concept_info(self, concept_name: str, select_first: bool = True) -> Tuple[int, str]:
#         """
#         Get information about a concept.
        
#         Args:
#             concept_name: The name of the concept to find
#             select_first: If True, automatically select the first match if multiple found
            
#         Returns:
#             Tuple of (concept_index, concept_label)
            
#         Raises:
#             ValueError: If no matching concepts found or multiple matches found and select_first is False
#         """
#         matching_indices = self.find_concept_indices(concept_name)
        
#         if not matching_indices:
#             raise ValueError(f"No concepts found with name '{concept_name}'")
        
#         if len(matching_indices) > 1:
#             print(f"Found multiple matching concepts:")
#             for idx in matching_indices:
#                 print(f"  {idx}: {self.mask_labels[idx]}")
            
#             if select_first:
#                 concept_idx = matching_indices[0]
#                 print(f"Using the first match: {self.mask_labels[concept_idx]}")
#             else:
#                 raise ValueError(f"Multiple concepts found with name '{concept_name}'. Set select_first=True to use the first match.")
#         else:
#             concept_idx = matching_indices[0]
        
#         return concept_idx, self.mask_labels[concept_idx]
    
#     def get_concept_sample_indices(self, concept_idx: int) -> np.ndarray:
#         """
#         Get the sample indices for a specific concept.
        
#         Args:
#             concept_idx: The index of the concept
            
#         Returns:
#             Array of sample indices where the concept is present
#         """
#         return np.where(self.mask_matrix[:, concept_idx] == 1)[0]
    
#     def get_average_contribution(self, concept_idx: int) -> np.ndarray:
#         """
#         Get the average contribution for a specific concept.
        
#         Args:
#             concept_idx: The index of the concept
            
#         Returns:
#             Average contribution vector across all samples for this concept
#         """
#         concept_indices = self.get_concept_sample_indices(concept_idx)
#         return np.mean(self.contributions[concept_indices], axis=0)

#     def get_average_contribution_channels(self, n: int = 10, concept_idx: int = None, method: str = 'argsort') -> np.ndarray:
#         """
#         Get important channels based on the average contribution of a specific concept.

#         Args:
#             n: Number of top channels to return (or number of std deviations if method='std')
#             concept_idx: The index of the concept
#             method: Method to select channels ('argsort' or 'std')
            
#         Returns:
#             Array of important channel indices
#         """
#         avg_contribution = self.get_average_contribution(concept_idx)

#         if method == 'argsort':
#             top_channels = np.argsort(avg_contribution)[-n:][::-1]
#         elif method == 'std':
#             mean = np.mean(avg_contribution)
#             std = np.std(avg_contribution)
#             threshold = mean + n * std
#             top_channels = np.where(avg_contribution > threshold)[0]
#         else:
#             raise ValueError(f"Unknown method: {method}. Use 'argsort' or 'std'.")

#         return list(top_channels)

#     def get_top_modes(self, concept_idx: int, n_modes: int = 5) -> np.ndarray:
#         """
#         Get the top modes (features) that are most correlated with a concept.
        
#         Args:
#             concept_idx: The index of the concept
#             n_modes: Number of top modes to return
            
#         Returns:
#             Array of indices of the top n_modes that correlate with the concept
#         """
#         # Sort modes by correlation with the concept
#         concept_modes = np.argsort(self.corr_mtx[:, concept_idx])
#         # Get top n_modes (in descending order)
#         top_modes = concept_modes[-n_modes:][::-1]
#         return top_modes
#     def get_channels(self, mode_indices: np.ndarray, n: int = 10, concept_idx=None,
#                         method: str = 'argsort') -> List[np.ndarray]:
#         """
#         Get the important channels for specified modes.
        
#         Args:
#             mode_indices: Array of mode indices to analyze
#             n: Number of top channels per mode to return. If None, returns all channels ranked by importance
#             concept_idx: Optional concept index (not used in this implementation)
#             method: Method to select channels ('argsort' or 'std')
            
#         Returns:
#             List of important channel indices across all modes
#         """
#         important_channels = []
#         for mode_idx in mode_indices:
#             mode = self.dictionary[mode_idx]
            
#             if method == 'argsort':
#                 if n is None:
#                     # Return all channels sorted by importance (highest to lowest)
#                     channels = np.argsort(mode)[::-1]  # Full sorted array in descending order
#                 else:
#                     # Get top n channels with highest values
#                     channels = np.argsort(mode)[-n:][::-1]
#             elif method == 'std':
#                 if n is None:
#                     # For std method with n=None, sort by deviation from mean (highest to lowest)
#                     mean = np.mean(mode)
#                     deviations = (mode - mean) / np.std(mode)
#                     channels = np.argsort(deviations)[::-1]  # Full sorted array in descending order
#                 else:
#                     # Get channels that are x standard deviations above the mean
#                     mean = np.mean(mode)
#                     std = np.std(mode)
#                     threshold = mean + n * std  # using n as the number of std devs
#                     channels = np.where(mode > threshold)[0]

#             elif method == 'cumsum':
#                 if n > 1.0:
#                     raise ValueError("For 'cumsum' method, n should be a float between 0 and 1.")
#                 elif n < 0.0:
#                     raise ValueError("For 'cumsum' method, n should be a float between 0 and 1.")
#                 channels = select_significant_indices(mode, method='threshold', param=n)
    
#             else:
#                 raise ValueError(f"Unknown method: {method}. Use 'argsort' or 'std' ")
                
#             important_channels.extend(channels)
#         important_channels = np.unique(important_channels)
#         return list(important_channels)
