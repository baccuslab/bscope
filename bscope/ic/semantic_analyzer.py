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

class SemanticAnalyzer:
    def __init__(self, semantic_hierarchy_path ='/data/codec/hierarchy_metadata/misc/semantic_indexes_test.json'):
        self.data = self.load_data(semantic_hierarchy_path)



    def load_data(self, path):
        """Load synset data from JSON file."""
        with open(path, 'r') as f:
            return json.load(f)

    def recursively_clean_names(self, tree):
        new_tree = {}
        
        # Reverse the name if it exists
        for k, v in tree.items():
            if k !='name':
                new_tree[k] = v
            else:
                new_tree['name'] = tree['name'].split('.n')[0]
        
        if 'children' in tree and tree['children']:
            new_tree['children'] = [self.recursively_clean_names(child) for child in tree['children']]

        return new_tree

    def add_level_depth(self, tree, current_level=0):
        """
        Recursively traverse a hierarchical tree and add a 'level' key 
        to each node indicating how many levels from the top it is.
        
        Args:
            tree (dict): Dictionary with keys 'name', 'definition', 'children'
            current_level (int): Current depth level (0 for root)
        
        Returns:
            dict: New tree with 'level' key added to each node
        """
        # Create a new dictionary to avoid modifying the original
        new_tree = {}

        for k, v in tree.items():
            if k != 'level':
                new_tree[k] = v
        
        # Add the level depth
        new_tree['level'] = current_level
        
        # Recursively process children with incremented level
        if 'children' in tree and tree['children']:
            new_tree['children'] = [
                self.add_level_depth(child, current_level + 1) 
                for child in tree['children']
            ]

        return new_tree
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
                if (term.decode('utf-8') if isinstance(term, bytes) else term) in info['path']:
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
        
    def get_normalized_distance(self, concept, from_top=True, use_global_max=False):
        """
        Get normalized distance (0-1) for a concept from top or bottom of the hierarchy.
        
        Parameters:
            concept (str): Concept name (e.g., 'dog')
            from_top (bool): If True, distance is measured from root to concept.
                            If False, distance is measured from concept to leaf.
            use_global_max (bool): If True, normalize against global max depth of hierarchy.
                                If False, normalize within concept's own branch.
        
        Returns:
            float or None: Normalized distance in [0, 1] or None if concept not found.
        """
        # Find the concept using same method as get_indices
        matches = self.indices_helper(concept, partial_match=True)

        if not matches:
            print(f"[Warning] Concept '{concept}' not found in semantic data.")
            return None

        # Choose the best match (you can modify sorting to prefer specific heuristics)
        concept_name = sorted(matches.keys(), key=lambda k: len(self.data[k]['path']))[0]
        concept_data = self.data[concept_name]

        # Get concept distances
        distance_from_root = concept_data.get('distance_from_root', 0)
        distance_to_leaves = concept_data.get('distance_to_leaves', 0)


        # Calculate the normalization factor
        if use_global_max:
            max_depth = 0
            for info in self.data.values():
                branch_depth = info.get('distance_from_root', 0) + info.get('distance_to_leaves', 0)
                max_depth = max(max_depth, branch_depth)
            normalizer = max_depth
        else:
            normalizer = distance_from_root + distance_to_leaves

        if normalizer <= 0:
            return 0.0

        if from_top:
            return distance_from_root / normalizer
        else:
            return distance_to_leaves / normalizer if not use_global_max else (normalizer - distance_from_root) / normalizer

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

            
if __name__ == "__main__":
    sem = SemanticAnalyzer()
    embed()




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
            else:
                raise ValueError(f"Unknown method: {method}. Use 'argsort' or 'std' ")
                
            important_channels.extend(channels)
            
        return list(set(important_channels))  # Return only unique channel indices
    def get_concept_correlations(self, concept_idx: int) -> np.ndarray:
        """
        Get correlations between a specific concept and all modes.
        
        Args:
            concept_idx: The index of the concept
            
        Returns:
            Array of correlations between the concept and each mode
        """
        return self.corr_mtx[:, concept_idx]
