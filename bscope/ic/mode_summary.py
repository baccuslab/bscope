import numpy as np
from IPython import embed
import h5py as h5
import matplotlib.pyplot as plt
import bscope
import bscope.ic as bic
from typing import List, Tuple, Dict, Union, Optional
from dataclasses import dataclass
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
    imgnet_corr_mtx: np.ndarray  # Correlation matrix with ImageNet classes

    loadings: np.ndarray
    dictionary: np.ndarray

    idx: Optional[int] = None  # Optional index for the layer, if needed
    r2: Optional[int] = None  # Optional R-squared values for the layer

    aggregated_data: Optional[np.ndarray] = None  # Aggregated data if available
    aggregated_reconstruction: Optional[np.ndarray] = None  # Aggregated reconstruction if available
    
    def __post_init__(self):
        self.corr_mtx[np.isnan(self.corr_mtx)] = 0  # Replace NaNs with zeros
        self.imgnet_corr_mtx[np.isnan(self.imgnet_corr_mtx)] = 0
        self.num_modes = self.dictionary.shape[0]


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

        # Load imgnet mask labels if available
        if 'imgnet_mask_labels' in self.file:
            raw_imgnet_labels = self.file['imgnet_mask_labels'][:]
            self.imgnet_mask_labels = [label.decode('utf-8').strip() if isinstance(label, bytes) 
                            else str(label).strip() for label in raw_imgnet_labels]
        else:
            self.imgnet_mask_labels = None


        self.mask_matrix = self.file['mask_matrix'][:] if 'mask_matrix' in self.file else None
        self.imgnet_mask_matrix = self.file['imgnet_mask_matrix'][:] if 'imgnet_mask_matrix' in self.file else None
        
        self.layer_idxs = np.sort([int(l) for l in list(self.file['layers'].keys())])
        self.layers = []
        for layer_idx in self.layer_idxs:
            layer_key = str(layer_idx)
            layer_data = self.file['layers'][layer_key]

            corr_mtx = layer_data['corr_mtx'][:]
            imgnet_corr_mtx = layer_data['imgnet_corr_mtx'][:] 
            corr_mtx=corr_mtx.T
            imgnet_corr_mtx=imgnet_corr_mtx.T
            loadings = layer_data['loadings'][:]
            dictionary = layer_data['dictionary'][:]
            aggregated_data = layer_data['data_agg'][:] if 'data' in layer_data else None
            aggregated_reconstruction = layer_data['reconstructed_agg'][:] 

            r2 = layer_data.attrs['r2'][()] if 'r2' in layer_data.attrs else None
            
            self.layers.append(LayerSummary(corr_mtx, imgnet_corr_mtx, loadings, dictionary, layer_idx, r2, aggregated_data, aggregated_reconstruction))

# @dataclass
# class LayerSummary:
#     """
#     A class to summarize the information of a single layer in the mode summary.
    
#     Attributes:
#         corr_mtx: Correlation matrix between loadings and mask matrix
#         r2: R-squared values for the layer
#         loadings: Loadings matrix from SAE (samples × features)
#         dictionary: Dictionary/atoms matrix from SAE (features × channels)
#     """
#     corr_mtx: np.ndarray

#     loadings: np.ndarray
#     dictionary: np.ndarray

#     idx: Optional[int] = None  # Optional index for the layer, if needed
#     r2: Optional[int] = None  # Optional R-squared values for the layer

#     aggregated_data: Optional[np.ndarray] = None  # Aggregated data if available
#     aggregated_reconstruction: Optional[np.ndarray] = None  # Aggregated reconstruction if available
    
#     def __post_init__(self):
#         self.corr_mtx[np.isnan(self.corr_mtx)] = 0  # Replace NaNs with zeros
#         self.num_modes = self.dictionary.shape[0]


# class ModeSummary:
#     def __init__(self, h5_path):
#         """
#         Initialize the ModeSummary with a path to an HDF5 file.
        
#         Args:
#             h5_path: Path to the HDF5 file containing mode summary data
#         """
#         self.h5_path = h5_path
#         self.file = h5.File(h5_path, 'r')
        
#         # Load mask labels if available
#         if 'mask_labels' in self.file:
#             raw_labels = self.file['mask_labels'][:]
#             # Convert bytes to strings, strip whitespace, and extract base name
#             self.mask_labels = [label.decode('utf-8').strip().split('.')[0] if isinstance(label, bytes) 
#                             else str(label).strip().split('.')[0] for label in raw_labels]
#         else:
#             self.mask_labels = None
#         self.mask_matrix = self.file['mask_matrix'][:] if 'mask_matrix' in self.file else None
        
#         self.layer_idxs = np.sort([int(l) for l in list(self.file['layers'].keys())])
#         self.layers = []
#         for layer_idx in self.layer_idxs:
#             layer_key = str(layer_idx)
#             layer_data = self.file['layers'][layer_key]

#             corr_mtx = layer_data['corr_mtx'][:]
#             corr_mtx=corr_mtx.T
#             loadings = layer_data['loadings'][:]
#             dictionary = layer_data['dictionary'][:]
#             aggregated_data = layer_data['data_agg'][:] if 'data' in layer_data else None
#             aggregated_reconstruction = layer_data['reconstructed_agg'][:] 

#             r2 = layer_data['r2'][()] if 'r2' in layer_data else None
            
#             self.layers.append(LayerSummary(corr_mtx, loadings, dictionary, layer_idx, r2, aggregated_data, aggregated_reconstruction))



# class ModeAnalyzer:
#     """
#     A class for analyzing modes and channels using ModeSummary data.
#     """
    

#     def __init__(self, mode_summary_path: ModeSummary):
#             """
#             Initialize the ModeAnalyzer with a ModeSummary instance.
            
#             Args:
#                 mode_summary: ModeSummary instance containing the data
#                 contributions: Optional contributions array for sample analysis
#             """
#             self.summary= ModeSummary(mode_summary_path)
    
#     def find_concept_indices(self, concept_name: str) -> List[int]:
#         """
#         Find the indices of concepts that match the given name.
        
#         Args:
#             concept_name: The name of the concept to find
            
#         Returns:
#             List of matching concept indices
#         """
#         if self.summary.mask_labels is None:
#             raise ValueError("No mask labels available in the ModeSummary")
            
#         matching_indices = []
#         concept_name_lower = concept_name.lower()
        
#         for i, label in enumerate(self.summary.mask_labels):
#             # Handle both string and bytes labels
#             if isinstance(label, bytes):
#                 label_str = label.decode('utf-8')
#             else:
#                 label_str = str(label)
            
#             # Extract base name (same as SemanticAnalyzer)
#             base_name = label_str.split('.')[0].strip().lower()
            
#             if base_name == concept_name_lower:
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
#                 print(f"  {idx}: {self.summary.mask_labels[idx]}")
            
#             if select_first:
#                 concept_idx = matching_indices[0]
#                 print(f"Using the first match: {self.summary.mask_labels[concept_idx]}")
#             else:
#                 raise ValueError(f"Multiple concepts found with name '{concept_name}'. Set select_first=True to use the first match.")
#         else:
#             concept_idx = matching_indices[0]
        
#         return concept_idx, self.summary.mask_labels[concept_idx]
    
#     def get_layer(self, layer_idx: int) -> LayerSummary:
#         """
#         Get a specific layer from the ModeSummary.
        
#         Args:
#             layer_idx: Index of the layer to retrieve
            
#         Returns:
#             LayerSummary for the specified layer
            
#         Raises:
#             ValueError: If layer_idx is not found
#         """
#         for layer in self.summary.layers:
#             if layer.idx == layer_idx:
#                 return layer
        
#         raise ValueError(f"Layer {layer_idx} not found. Available layers: {self.summary.layer_idxs}")
    
#     def get_top_modes(self, layer_idx: int, concept_name: str, method: str = 'percentile', 
#                      param: float = 0.7, min_indices: int = 1, max_indices: int = 50,
#                      select_first: bool = True) -> np.ndarray:
#         """
#         Get top modes for a specific layer and concept using select_significant_indices.
        
#         Args:
#             layer_idx: Index of the layer
#             concept_name: Name of the concept to analyze
#             method: Method for select_significant_indices ('threshold', 'percentile', etc.)
#             param: Parameter for the selection method
#             min_indices: Minimum number of indices to return
#             max_indices: Maximum number of indices to return
#             select_first: Whether to auto-select first concept match
            
#         Returns:
#             Array of mode indices
#         """
#         # Get the layer
#         layer = self.get_layer(layer_idx)
        
#         # Get concept info
#         concept_idx, concept_label = self.get_concept_info(concept_name, select_first)
        
#         # Get correlations for this concept (equivalent to your rs array)
#         correlations = layer.corr_mtx[:, concept_idx]
        
#         # Use select_significant_indices to get top modes
#         modes = select_significant_indices(
#             correlations, 
#             method=method, 
#             param=param, 
#             min_indices=min_indices, 
#             max_indices=max_indices
#         )
        
#         return modes
    
#     def get_top_channels(self, layer_idx: int, concept_name: str, 
#                         mode_method: str = 'percentile', mode_param: float = 0.7,
#                         mode_min_indices: int = 1, mode_max_indices: int = 50,
#                         channel_method: str = 'percentile', channel_param: float = 0.5,
#                         channel_min_indices: int = 1, channel_max_indices: int = 50,
#                         select_first: bool = True, concat: bool = False) -> List[int]:
#         """
#         Get top channels for a specific layer and concept by first getting top modes,
#         then getting channels from those modes' dictionary vectors.
        
#         Args:
#             layer_idx: Index of the layer
#             concept_name: Name of the concept to analyze
#             mode_method: Method for selecting modes
#             mode_param: Parameter for mode selection
#             mode_min_indices: Minimum number of modes
#             mode_max_indices: Maximum number of modes
#             channel_method: Method for selecting channels from each mode
#             channel_param: Parameter for channel selection
#             channel_min_indices: Minimum number of channels per mode
#             channel_max_indices: Maximum number of channels per mode
#             select_first: Whether to auto-select first concept match
#             concat: If True, map doubled channel indices back to original channel space
            
#         Returns:
#             List of unique channel indices across all selected modes
#         """
#         # Get the layer
#         layer = self.get_layer(layer_idx)
        
#         # Get top modes first
#         top_modes = self.get_top_modes(
#             layer_idx, concept_name, mode_method, mode_param,
#             mode_min_indices, mode_max_indices, select_first
#         )
        
#         # Collect channels from all modes
#         all_channels = []
        
#         for mode_idx in top_modes:
#             # Get the dictionary vector (atom) for this mode
#             atom = layer.dictionary[mode_idx, :]

            
#             # Get top channels for this atom
#             top_channels = select_significant_indices( 
#                 atom,
#                 method=channel_method,
#                 param=channel_param,
#                 min_indices=channel_min_indices,
#                 max_indices=channel_max_indices
#             )
            
#             all_channels.extend(top_channels)
        
#         # Determine which channels to return
#         if len(top_modes) == 1:
#             channels = top_channels
#         else: 
#             channels = all_channels
        
#         # Apply concat mapping if requested
#         if concat:
#             channels = self.map_to_original_channels(channels, layer)
        
#         return channels

    
#     def get_mode_correlations(self, layer_idx: int, concept_name: str, 
#                              select_first: bool = True) -> np.ndarray:
#         """
#         Get correlation values for all modes with a specific concept.
        
#         Args:
#             layer_idx: Index of the layer
#             concept_name: Name of the concept
#             select_first: Whether to auto-select first concept match
            
#         Returns:
#             Array of correlation values (equivalent to your rs array)
#         """
#         layer = self.get_layer(layer_idx)
#         concept_idx, _ = self.get_concept_info(concept_name, select_first)
#         return layer.corr_mtx[:, concept_idx]
    
#     def get_concept_sample_indices(self, concept_idx: int) -> np.ndarray:
#         """
#         Get the sample indices for a specific concept.
        
#         Args:
#             concept_idx: The index of the concept
            
#         Returns:
#             Array of sample indices where the concept is present
#         """
#         return np.where(self.mask_matrix[:, concept_idx] == 1)[0]

#     def find_similar_concepts_by_channels(
#         self, 
#         seed_concept: str, 
#         layer_idx: int,
#         mode_method: str = 'percentile', 
#         mode_param: float = 0.7,
#         channel_method: str = 'std',  # Changed default to 'std'
#         channel_param: float = 2.0,   # Changed default to 2.0 std deviations
#         min_overlap: int = 1,
#         select_first: bool = True,
#         concat=False,
#         # Remove top_n_channels parameter to avoid limiting
#     ):
#         """
#         Find concepts that share the most contributing channels with a seed concept,
#         using consistent channel selection criteria for all concepts.
        
#         Args:
#             seed_concept: The concept to find similar concepts for
#             layer_idx: Which layer to analyze
#             mode_method: Method for selecting the most salient mode
#             mode_param: Parameter for mode selection
#             channel_method: Method for selecting top channels ('percentile', 'std', 'threshold', etc.)
#             channel_param: Parameter for channel selection (percentile value, std deviations, or threshold)
#             min_overlap: Minimum number of shared channels to include in results
#             select_first: Whether to auto-select first concept match
                    
#         Returns:
#             List of tuples: (concept_name, shared_count, overlap_ratio, shared_channels)
#             Sorted by number of shared channels (descending)
#         """
        
#         print(f"Finding concepts similar to '{seed_concept}' at layer {layer_idx}")
#         print(f"Using channel selection method: {channel_method}, param: {channel_param}")
#         print("-" * 60)
        
#         # Get seed concept's channels using the specified method
#         try:
#             seed_channels = self.get_top_channels(
#                 layer_idx=layer_idx,
#                 concept_name=seed_concept,
#                 mode_method=mode_method,
#                 mode_param=mode_param,
#                 mode_min_indices=1,
#                 mode_max_indices=1,  # Just get the most salient mode
#                 channel_method=channel_method,
#                 channel_param=channel_param,
#                 channel_min_indices=1,             # Minimum of 1 channel
#                 channel_max_indices=float('inf'),  # No upper limit - important!
#                 select_first=select_first,
#                 concat=concat,
#             )
            
#             # No limit on top_n_channels anymore
#             seed_channels_list = seed_channels  # Keep ordered list
#             seed_channels_set = set(seed_channels_list)  # Create set for intersections
            
#         except Exception as e:
#             print(f"Error getting channels for seed concept '{seed_concept}': {e}")
#             return []
        
#         print(f"Seed concept '{seed_concept}' selected {len(seed_channels)} channels: {seed_channels}")
        
#         # Compare with all other concepts
#         results = []
        
#         # Get all concepts to check
#         syn = bic.SemanticAnalyzer('/home/zalaoui/semantic_indexes_test.json')
#         _, imagenet_class_names = syn.get_all_imagenet_masks(list(range(1000)))
        
#         for concept_label in imagenet_class_names:
#             # Skip the seed concept itself
#             base_concept_name = concept_label.split('.')[0]
#             if base_concept_name.lower() == seed_concept.lower():
#                 continue
                
#             try:
#                 # Get this concept's channels using EXACTLY THE SAME method and parameters
#                 concept_channels = self.get_top_channels(
#                     layer_idx=layer_idx,
#                     concept_name=base_concept_name,
#                     mode_method=mode_method,
#                     mode_param=mode_param,
#                     mode_min_indices=1,
#                     mode_max_indices=1,  # Just get the most salient mode
#                     channel_method=channel_method,
#                     channel_param=channel_param,
#                     channel_min_indices=1,             # Minimum of 1 channel
#                     channel_max_indices=float('inf'),  # No upper limit - important!
#                     select_first=True, # Auto-select to avoid prompts
#                     concat=concat
#                 )
                
#                 # No limit on top_n_channels anymore
#                 concept_channels_set = set(concept_channels)
                
#                 # Calculate overlap
#                 shared_channels = seed_channels_set.intersection(concept_channels_set)
#                 shared_count = len(shared_channels)
                
#                 # Calculate overlap ratio based on seed channels
#                 total_channels = len(seed_channels_set)
#                 overlap_ratio = shared_count / total_channels if total_channels > 0 else 0
                
#                 # Only include if meets minimum overlap threshold
#                 if shared_count >= min_overlap:
#                     results.append((
#                         base_concept_name,
#                         shared_count,
#                         overlap_ratio,
#                         sorted(list(shared_channels))
#                     ))
                    
#             except Exception as e:
#                 # Skip concepts that cause errors (e.g., not found)
#                 continue
        
#         # Sort by shared count (descending), then by overlap ratio
#         results.sort(key=lambda x: (x[1], x[2]), reverse=True)
#         print("=" * 80)
#         print(f"Seed concept '{seed_concept}' selected {len(seed_channels)} channels: {list(seed_channels)}")
#         return results
#     def print_similar_concepts(
#         self, 
#         results: List[Tuple[str, int, float, List[int]]], 
#         top_n: int = 10,
#         show_channels: bool = True
#     ):
#         """
#         Pretty print the results from find_similar_concepts_by_channels
        
#         Args:
#             results: Output from find_similar_concepts_by_channels
#             top_n: Number of top results to show
#             show_channels: Whether to show the actual shared channel numbers
#         """
        
#         if not results:
#             print("No similar concepts found.")
#             return
            
#         print("Most Similar Concepts:")
#         print("=" * 80)
        
#         for i, (concept, shared_count, overlap_ratio, shared_channels) in enumerate(results[:top_n]):
#             if concept == 'black_grouse':
#                 print("HOLY BLACK GROUSE👀👀👀")

#             print(f"{i+1:2d}. {concept:20s} | "
#                   f"Shared: {shared_count:2d} | "
#                   f"Overlap: {overlap_ratio:.1%}")
            
#             if show_channels and shared_channels:
#                 # Show channels in groups of 10 for readability
#                 channel_str = str(shared_channels)
#                 if len(channel_str) > 80:
#                     channel_str = channel_str[:77] + "..."
#                 print(f"     Shared channels: {channel_str}")
#             print()
    
#     def plot_mode_comparison(
#             self,
#             seed_concept: str,
#             results: List[Tuple[str, int, float, List[int]]],
#             layer_idx: int,
#             top_n_display: int = 10,
#             mode_method: str = 'top_n',
#             mode_param: int = 1,
#             figsize_per_subplot: Tuple[int, int] = (12, 3),
#             # Remove seed_top_n_channels parameter
#             channel_method: str = 'std',  # Add these parameters to be consistent
#             channel_param: float = 2.0,
#             concat=False,    # with find_similar_concepts_by_channels,
#             text=False 
#         ):
#         """
#         Plot seed concept's top mode and similar concepts' top modes with shared channels highlighted.
        
#         Args:
#             analyzer: ModeAnalyzer instance
#             seed_concept: The original seed concept
#             results: Output from find_similar_concepts_by_channels
#             layer_idx: Layer to analyze
#             top_n_display: Number of similar concepts to display
#             mode_method: Method for getting top mode
#             mode_param: Parameter for mode selection
#             figsize_per_subplot: Size of each subplot
#             channel_method: Method for selecting channels (same as in find_similar_concepts)
#             channel_param: Parameter for channel selection (same as in find_similar_concepts)
#         """
    

#         display_results = results[:top_n_display]
#         total_concepts = len(display_results) + 1  # +1 for seed
        
#         # Create subplot grid
#         fig, axes = plt.subplots(total_concepts, 1, 
#                                 figsize=(figsize_per_subplot[0], 
#                                     figsize_per_subplot[1] * total_concepts))
        
#         if total_concepts == 1:
#             axes = [axes]
        
#         # Get layer for extracting dictionary atoms
#         layer = self.get_layer(layer_idx)
        
#         # 1. Plot SEED CONCEPT first
#         print(f"Getting seed concept '{seed_concept}' top mode...")
#         seed_modes = self.get_top_modes(
#             layer_idx=layer_idx,
#             concept_name=seed_concept,
#             method=mode_method,
#             param=mode_param,
#             min_indices=1,
#             max_indices=1
#         )
        
#         seed_mode_idx = seed_modes[0]
#         seed_atom = layer.dictionary[seed_mode_idx, :]
#         seed_correlations = self.get_mode_correlations(layer_idx=layer_idx, concept_name=seed_concept)
        
#         # Get seed's top channels using the SAME method as in find_similar_concepts
#         seed_top_channels = self.get_top_channels(
#             layer_idx=layer_idx,
#             concept_name=seed_concept,
#             mode_method=mode_method,
#             mode_param=mode_param,
#             mode_min_indices=1,
#             mode_max_indices=1,
#             channel_method=channel_method,
#             channel_param=channel_param,
#             channel_min_indices=1,
#             channel_max_indices=float('inf'),
#             select_first=True,
#             concat=concat
#         )
#         seed_top_channels_set = set(seed_top_channels)
        
#         # Plot seed concept
#         ax = axes[0]
#         ax.plot(seed_atom, 'k-', linewidth=1, alpha=0.8)
#         ax.set_title(f'SEED: {seed_concept} (mode {seed_mode_idx}, corr={seed_correlations[seed_mode_idx]:.3f})', 
#                     fontsize=8, fontweight='bold', color='blue')
#         ax.set_ylabel('Activation', fontsize=12)

#         if text:
        
#             for ch in seed_top_channels_set:
#                 ax.axvline(x=ch, color='blue', linestyle='-', alpha=0.2, linewidth=1.5)
            
#             ax.text(0.02, 0.95, f'{len(seed_top_channels_set)} top channels', 
#                     transform=ax.transAxes, va='top', fontsize=10, 
#                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3))
        
#         # 2. Plot SIMILAR CONCEPTS
#         for i, (concept_name, shared_count, overlap_ratio, shared_channels) in enumerate(display_results):
#             ax = axes[i + 1]
            
#             # Get this concept's top mode
#             try:
#                 # Extract base name if needed
#                 base_concept_name = concept_name.split('.')[0]
                
#                 concept_modes = self.get_top_modes(
#                     layer_idx=layer_idx,
#                     concept_name=base_concept_name,
#                     method=mode_method,
#                     param=mode_param,
#                     min_indices=1,
#                     max_indices=1,
#                     select_first=True
#                 )
                
#                 concept_mode_idx = concept_modes[0]
#                 concept_atom = layer.dictionary[concept_mode_idx, :]
#                 concept_correlations = self.get_mode_correlations(
#                     layer_idx=layer_idx, concept_name=base_concept_name, select_first=True
#                 )
                
#                 # Plot the atom
#                 ax.plot(concept_atom, 'k-', linewidth=1, alpha=0.8)
#                 ax.set_title(f'{concept_name} (mode {concept_mode_idx}, corr={concept_correlations[concept_mode_idx]:.3f})\n'
#                             f'Shared: {shared_count}/{len(seed_top_channels_set)} ({overlap_ratio:.1%})', 
#                             fontsize=8)
#                 ax.set_ylabel('Activation', fontsize=8)

#                 if text:
                
#                     # Highlight shared channels in RED
#                     for ch in shared_channels:
#                         ax.axvline(x=ch, color='red', linestyle='-', alpha=0.2, linewidth=2)
                    
#                     # Highlight seed's non-shared top channels in light blue
#                     non_shared_seed_channels = seed_top_channels_set - set(shared_channels)
#                     for ch in non_shared_seed_channels:
#                         ax.axvline(x=ch, color='lightblue', linestyle='--', alpha=0.4, linewidth=1)
                    
#                     # Add legend info
#                     ax.text(0.02, 0.95, f'{len(shared_channels)} shared channels', 
#                             transform=ax.transAxes, va='top', fontsize=10,
#                             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
                
#             except Exception as e:
#                 ax.text(0.5, 0.5, f'Error loading {concept_name}:\n{str(e)}', 
#                         transform=ax.transAxes, ha='center', va='center', fontsize=10)
#                 ax.set_title(f'{concept_name} (ERROR)', fontsize=12, color='red')
        
#         # Set x-label only on bottom plot
#         axes[-1].set_xlabel('Channel Index', fontsize=12)
#         # No x-tick labels
#         for i, ax in enumerate(axes):
#             if i != len(axes) - 1:
#                 ax.set_xticks([])
        
#         # Add overall title and legend
#         fig.suptitle(f'Mode Comparison: {seed_concept} vs Similar Concepts (Layer {layer_idx})', 
#                     fontsize=8, fontweight='bold')
        
#         # Create legend
#         if text:
#             from matplotlib.lines import Line2D
#             legend_elements = [
#                 Line2D([0], [0], color='blue', lw=2, label=f'{seed_concept} top channels'),
#                 Line2D([0], [0], color='red', lw=2, label='Shared channels'),
#                 Line2D([0], [0], color='lightblue', lw=1, linestyle='--', alpha=0.6, 
#                     label=f'{seed_concept} non-shared channels')
#             ]
#             fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
#         plt.tight_layout()
#         plt.subplots_adjust(top=0.92)  # Make room for suptitle and legend
#         plt.subplots_adjust(hspace=0.5)  # Increase vertical spacing
#         plt.show()
        
#         return fig

#     def get_concepts_with_shared_channels(
#     self, 
#     channels: List[int], 
#     layer_idx: int, 
#     top_n: int = 5,
#     concat: bool= False,
#     min_overlap: int = 1
#     ) -> List[Tuple[str, int, List[int]]]:
#             """
#             Find concepts that have the given channels in their top channels list.
            
#             Args:
#                 channels: List of channel indices to search for
#                 layer_idx: Layer to analyze
#                 top_n: Number of top channels to get for each concept
#                 min_overlap: Minimum number of shared channels to include
                
#             Returns:
#                 List of (concept_name, shared_count, shared_channels) sorted by shared_count
#             """
#             results = []
#             channels_set = set(channels)
            
#             for concept_label in self.summary.mask_labels:
#                 try:
#                     # Get this concept's top channels
#                     concept_channels = self.get_top_channels(
#                         layer_idx=layer_idx,
#                         concept_name=concept_label,
#                         mode_method='top_n',
#                         mode_param=1,
#                         mode_min_indices=1,
#                         mode_max_indices=1,
#                         channel_method='top_n',
#                         channel_param=top_n,
#                         channel_min_indices=top_n,
#                         channel_max_indices=top_n,
#                         select_first=True,
#                         concat=concat
#                     )
                    
#                     concept_channels_set = set(concept_channels[:top_n])
#                     shared_channels = list(channels_set.intersection(concept_channels_set))
#                     shared_count = len(shared_channels)
                    
#                     if shared_count >= min_overlap:
#                         results.append((concept_label, shared_count, shared_channels))
                        
#                 except Exception as e:
#                     # Skip concepts that cause errors
#                     continue
            
#             # Sort by shared count (descending)
#             results.sort(key=lambda x: x[1], reverse=True)
#             return results
#     def map_to_original_channels(self, doubled_indices, layer):
#         """Convert doubled channel indices back to original channel indices"""
#         original_n_channels = layer.dictionary.shape[1] // 2
        
#         pos_channels = []
#         neg_channels = []
        
#         for idx in doubled_indices:
#             if idx < original_n_channels:
#                 pos_channels.append(idx)
#             else:
#                 neg_channels.append(idx - original_n_channels)
        
#         # Check for conflicts and print if found
#         pos_set = set(pos_channels)
#         neg_set = set(neg_channels)
#         overlap = pos_set.intersection(neg_set)
        
#         print(f"Found {len(overlap)} channels in both positive and negative: {sorted(overlap)}")
        
#         return list(np.unique(pos_channels + neg_channels))
#     def discover_related_concepts(
#         self,
#         seed_concept: str,
#         layer_idx: int,
#         seed_top_channels: int = 5,
#         concepts_top_channels: int = 5,
#         min_overlap: int = 1,
#         mode_method: str = 'top_n',
#         mode_param: int = 1,
#         channel_method: str = 'top_n',
#         channel_param: int = 10,
#         select_first: bool = True
#     ) -> Tuple[List[str], List[int], Dict]:
#         """
#         Discover related concepts through direct channel overlap analysis.

#         Args:
#             seed_concept: Starting concept
#             layer_idx: Layer to analyze
#             seed_top_channels: Number of top channels to get from seed
#             concepts_top_channels: Number of top channels to consider for each concept
#             min_overlap: Minimum number of shared channels to include
#             ... (other params same as existing methods)

#         Returns:
#             Tuple of (all_concepts, all_channels, discovery_info)
#         """

#         print(f"Starting discovery from seed concept: '{seed_concept}'")
#         print("=" * 60)

#         # Step 1: Get seed concept's top channels
#         print(f"Step 1: Getting top {seed_top_channels} channels for '{seed_concept}'")

#         seed_channels = self.get_top_channels(
#             layer_idx=layer_idx,
#             concept_name=seed_concept,
#             mode_method=mode_method,
#             mode_param=mode_param,
#             mode_min_indices=1,
#             mode_max_indices=1,
#             channel_method=channel_method,
#             channel_param=channel_param,
#             channel_min_indices=seed_top_channels,
#             channel_max_indices=seed_top_channels,
#             select_first=select_first
#         )

#         seed_channels = seed_channels[:seed_top_channels]
#         print(f"Seed channels: {seed_channels}")
#         print()

#         # Step 2: Find concepts that share these channels
#         print(f"Step 2: Finding concepts that have these channels in their top {concepts_top_channels}")

#         shared_concepts = self.get_concepts_with_shared_channels(
#             channels=seed_channels,
#             layer_idx=layer_idx,
#             top_n=concepts_top_channels,
#             min_overlap=min_overlap
#         )

#         # Remove seed concept from results and print
#         discovered_concepts = []
#         for concept_name, shared_count, shared_channels_list in shared_concepts:
#             if concept_name.lower() != seed_concept.lower():
#                 discovered_concepts.append(concept_name)
#                 print(f"{concept_name}: {shared_count} shared channels {shared_channels_list}")

#         print(f"\nDiscovered {len(discovered_concepts)} related concepts")
#         print()

#         # Step 3: Get top channels for each discovered concept
#         print(f"Step 3: Getting top {seed_top_channels} channels for each discovered concept")

#         all_channels = set(seed_channels)
#         concept_channel_map = {seed_concept: seed_channels}

#         for concept in discovered_concepts:
#             try:
#                 concept_channels = self.get_top_channels(
#                     layer_idx=layer_idx,
#                     concept_name=concept,
#                     mode_method=mode_method,
#                     mode_param=mode_param,
#                     mode_min_indices=1,
#                     mode_max_indices=1,
#                     channel_method=channel_method,
#                     channel_param=channel_param,
#                     channel_min_indices=seed_top_channels,
#                     channel_max_indices=seed_top_channels,
#                     select_first=True,
#                 )

#                 concept_channels = concept_channels[:seed_top_channels]
#                 concept_channel_map[concept] = concept_channels
#                 all_channels.update(concept_channels)

#                 print(f"{concept}: {concept_channels}")

#             except Exception as e:
#                 print(f"Error getting channels for {concept}: {e}")
#                 continue

#         all_channels = sorted(list(all_channels))
#         all_concepts = [seed_concept] + discovered_concepts

#         print()
#         print("=" * 60)
#         print("DISCOVERY SUMMARY")
#         print("=" * 60)
#         print(f"Seed concept: {seed_concept}")
#         print(f"Discovered concepts ({len(discovered_concepts)}): {discovered_concepts}")
#         print(f"Total concepts: {len(all_concepts)}")
#         print(f"Total unique channels: {len(all_channels)}")
#         print(f"All channels: {all_channels}")

#         # Package discovery info
#         discovery_info = {
#             'seed_concept': seed_concept,
#             'seed_channels': seed_channels,
#             'shared_concepts': shared_concepts,
#             'concept_channel_map': concept_channel_map,
#             'discovered_concepts': discovered_concepts
#         }

#         return all_concepts, all_channels, discovery_info

#     def print_discovery_network(self, discovery_info: Dict):
#         """
#         Pretty print the discovery network showing connections.
#         """
#         print("\nDISCOVERY NETWORK")
#         print("=" * 50)

#         seed = discovery_info['seed_concept']
#         seed_channels = discovery_info['seed_channels']
#         shared_concepts = discovery_info['shared_concepts']

#         print(f"🌱 SEED: {seed}")
#         print(f"   Channels: {seed_channels}")
#         print()

#         print("🔗 CONCEPTS WITH SHARED CHANNELS:")
#         for concept_name, shared_count, shared_channels_list in shared_concepts:
#             if concept_name.lower() != seed.lower():
#                 print(f"   {concept_name}: {shared_count} shared {shared_channels_list}")

#         print("\n📊 DISCOVERED CONCEPT CHANNELS:")
#         concept_channel_map = discovery_info['concept_channel_map']
#         for concept, channels in concept_channel_map.items():
#             if concept != seed:
#                 print(f"   {concept}: {channels}")

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
   

# Add these methods to your ModeAnalyzer class by copying them in



# Add these methods to your ModeAnalyzer class by copying them in

# Add these methods to your ModeAnalyzer class by copying them in

    # def get_average_contribution(self, concept_name: str, select_first: bool = True) -> np.ndarray:
    #     """
    #     Get the average contribution for a specific concept.
        
    #     Args:
    #         concept_name: The name of the concept
    #         select_first: Whether to auto-select first concept match
            
    #     Returns:
    #         Average contribution vector across all samples for this concept
    #     """
    #     # Get concept index from name
    #     concept_idx, _ = self.get_concept_info(concept_name, select_first)
        
    #     # Get sample indices for this concept
    #     concept_indices = self.get_concept_sample_indices(concept_idx)
        
    #     # Return average contribution across samples
    #     return np.mean(self.summary[concept_indices], axis=0)

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
