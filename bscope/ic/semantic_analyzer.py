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
    def __init__(self, semantic_hierarchy_path ='/home/zalaoui/semantic_indexes_test.json'):
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

def get_top_mode(mode_summary, layer, class_idx, which_mode=1):
    corrs = mode_summary.layers[layer].imgnet_corr_mtx
    top_mode = np.argsort(corrs[:, class_idx])[::-1]
    top_mode = top_mode[which_mode]
    
    atom = mode_summary.layers[layer].dictionary[top_mode]
    loadings = mode_summary.layers[layer].loadings[:, top_mode]
    corr = corrs[top_mode, class_idx]
    return top_mode, atom, loadings, corr

def single_image_semantic_loading(mode_summary, layer, image_idx):
    corrs = mode_summary.layers[layer].imgnet_corr_mtx
    loadings = mode_summary.layers[layer].loadings[image_idx]
    loading_idxs = np.where(loadings > 0.5)[0]
    imagenet_labels = mode_summary.imgnet_mask_labels
    labels = []
    for loading_idx in loading_idxs: 
        # Classes
        corr = corrs[loading_idx,:]
        top_classes = np.argsort(corr)[::-1]
        top_class = top_classes[0]
        imagenet_label = imagenet_labels[top_class]
        labels.append(imagenet_label)


    return loadings, loading_idxs, labels

def top_n(vector, n=5):
    idxs = np.argsort(vector)[-n:][::-1]
    return idxs, vector[idxs]

def load_hierarchy(path='/data/hierarchy_metadata/pruned_hierarchy.json'):
    with open(path, 'r') as f:
        return json.load(f)

def get_masks(path='/data/hierarchy_metadata/pruned_hierarchy.json', leaf_only=False, targets=None):
    if targets is None:
        targets = []
        for i in range(1000):
            targets.extend(np.ones(50)*i)
        targets = np.array(targets).astype(int)

    print("Loading hierarchy from:", path)
    hierarchy = load_hierarchy(path)
    
    masks = []
    labels = []
    for k,v in hierarchy.items():
        idxs = v['idxs']

        if leaf_only and not v['leaf']:
            continue

        masks.append(np.isin(targets, idxs))
        labels.append(k)

    return np.array(masks), labels

def chunk_masks(mask_matrix, bins = [50, 5000]):

    if bins[0] != 0:
        bins = [0] + bins

    if bins[-1] != None:
        bins = bins + [np.inf]

    summed = mask_matrix.sum(1)
    
    valid_idxs = []
    for b1, b2 in zip(bins[:-1], bins[1:]):
        valid = np.where((summed > b1) & (summed <= b2))[0]
        valid_idxs.append(valid)

    return valid_idxs

    # if target_indices is None:
    #     target_indices = list(range(1000))
    
    # data = load_hierarchy(path)
    # node_names = list(data.keys())
    # num_nodes = len(node_names)

    # masks = []

    # for node in node_names:
    #     indices = set(data[node]['idxs'])
    #     mask = np.array([idx in indices for idx in target_indices])
    #     masks.append(mask)

    # mask_array = np.array(masks)
    # return mask_array, node_names

# class SemanticAnalyzer:
#     def __init__(self, semantic_hierarchy_path ='/data/codec/hierarchy_metadata/misc/semantic_indexes_test.json'):
#         self.data = self.load_data(semantic_hierarchy_path)



#     def load_data(self, path):
#         """Load synset data from JSON file."""
#         with open(path, 'r') as f:
#             return json.load(f)

#     def recursively_clean_names(self, tree):
#         new_tree = {}
        
#         # Reverse the name if it exists
#         for k, v in tree.items():
#             if k !='name':
#                 new_tree[k] = v
#             else:
#                 new_tree['name'] = tree['name'].split('.n')[0]
        
#         if 'children' in tree and tree['children']:
#             new_tree['children'] = [self.recursively_clean_names(child) for child in tree['children']]

#         return new_tree

#     def add_level_depth(self, tree, current_level=0):
#         """
#         Recursively traverse a hierarchical tree and add a 'level' key 
#         to each node indicating how many levels from the top it is.
        
#         Args:
#             tree (dict): Dictionary with keys 'name', 'definition', 'children'
#             current_level (int): Current depth level (0 for root)
        
#         Returns:
#             dict: New tree with 'level' key added to each node
#         """
#         # Create a new dictionary to avoid modifying the original
#         new_tree = {}

#         for k, v in tree.items():
#             if k != 'level':
#                 new_tree[k] = v
        
#         # Add the level depth
#         new_tree['level'] = current_level
        
#         # Recursively process children with incremented level
#         if 'children' in tree and tree['children']:
#             new_tree['children'] = [
#                 self.add_level_depth(child, current_level + 1) 
#                 for child in tree['children']
#             ]

#         return new_tree
#     def indices_helper(self, name, partial_match=False):
#         """
#         Retrieve synsets by name or partial match, including all indices from descendants.
#         """
#         # Get all descendants using the get_descendant_nodes function
#         all_descendants = self.get_descendant_nodes(name)

#         # If no descendants found, try with partial match
#         if not all_descendants and partial_match:
#             # Try again with partial matching
#             direct_matches = {}
#             for synset_name, info in self.data.items():
#                 base_name = synset_name.split('.')[0]
#                 if name.lower() in base_name.lower():
#                     direct_matches[synset_name] = info

#             # Return just these direct partial matches if no descendants found
#             if direct_matches:
#                 return direct_matches

#         # Process the descendants to add combined indices to each direct match
#         results = {}
#         direct_matches_found = False

#         # First identify direct matches
#         for synset_name, info in all_descendants.items():
#             base_name = synset_name.split('.')[0]
#             if base_name.lower() == name.lower():
#                 # This is a direct match - store it separately
#                 results[synset_name] = info.copy(
#                 )  # Copy to avoid modifying original
#                 direct_matches_found = True

#         # If no direct matches but we have descendants, include all descendants
#         if not direct_matches_found and all_descendants:
#             # Just return all descendants when no direct matches
#             return all_descendants

#         # Now, let's add all indices from all descendants to each direct match
#         if direct_matches_found:
#             # Collect all indices from all descendants
#             all_indices = []
#             for desc_name, desc_info in all_descendants.items():
#                 # Get indices from this descendant
#                 if 'indices' in desc_info and desc_info['indices']:
#                     all_indices.extend(desc_info['indices'])

#             # Remove duplicates and sort
#             all_indices = sorted(list(set(all_indices)))

#             # Add these combined indices to each direct match
#             for match_name in results:
#                 results[match_name]['all_descendant_indices'] = all_indices

#         # Return the results
#         return results

#     def get_indices(self, name, partial_match=False):
#         """
#         Print matching synsets and also show all indices from all descendants.
#         Returns a list of all unique indices from all descendants.
#         """
#         # First get direct matches
#         matches = self.indices_helper(name, partial_match)

#         if not matches:
#             print(f"No synsets found with name '{name}'.")
#             return []

#         print(f"Found {len(matches)} matching term for '{name}':")

#         all_descendants = self.get_descendant_nodes(name)

#         # Collect all indices from all descendants
#         all_indices = set()
#         for desc_name, desc_info in all_descendants.items():
#             if 'indices' in desc_info and desc_info['indices']:
#                 all_indices.update(desc_info['indices'])

#         # Sort the indices
#         all_indices = sorted(list(all_indices))


#         # Return all unique indices for use in mask function
#         return all_indices

#     def get_parent_nodes(self, name, partial_match=False):
#         """Find parent nodes for the given name."""
#         matches = self.indices_helper(name, partial_match)
#         if not matches:
#             print(f"No synsets found with name '{name}'.")
#             return {}
#         parent_nodes = {}
#         for synset_name, info in matches.items():
#             full_path = info['path']
#             components = full_path.split('.')
#             current_path = ""
#             for i, component in enumerate(components[:-1]):
#                 current_path += ("." if i > 0 else "") + component
#                 for other_name, other_info in self.data.items():
#                     if other_info['path'] == current_path:
#                         parent_nodes[other_name] = other_info
#         return parent_nodes

#     def get_descendant_nodes(self, term):
#         """
#         Find all synsets that are descendants of or match the given term.
#         The term is matched against the name part of the synset.
#         """
#         descendant_nodes = {}
#         matched_synsets = []

#         # First try to find direct matches
#         for synset_name, info in self.data.items():
#             base_name = synset_name.split('.')[0]
#             if base_name.lower() == term.lower():
#                 descendant_nodes[synset_name] = info
#                 matched_synsets.append(synset_name)

#         # If we found direct matches, look for their descendants
#         for matched_synset in matched_synsets:
#             for synset_name, info in self.data.items():
#                 if synset_name in descendant_nodes:
#                     continue  # Skip if already added
#                 # Check if the synset is a descendant of any matched synset
#                 if matched_synset in info['path']:
#                     descendant_nodes[synset_name] = info

#         # If we still didn't find anything or if the term is something like 'aquatic_mammal',
#         # try looking for it as a component in paths
#         if not descendant_nodes:
#             for synset_name, info in self.data.items():
#                 if (term.decode('utf-8') if isinstance(term, bytes) else term) in info['path']:
#                     descendant_nodes[synset_name] = info
#                     # Now look for descendants of this synset
#                     for other_synset, other_info in self.data.items():
#                         if synset_name in other_info[
#                                 'path'] and other_synset != synset_name:
#                             descendant_nodes[other_synset] = other_info

#         return descendant_nodes


#     def get_all_indices_for_search(self, term):
#         """
#         Get ALL indices from all synsets that have the term in their name or path.
#         This includes all direct and indirect descendants.
#         """
#         all_indices = []

#         # Get all descendant nodes
#         descendants = self.get_descendant_nodes(term)

#         # Extract all indices from these descendants
#         for info in descendants.values():
#             all_indices.extend(info.get('indices', []))

#         return sorted(list(set(all_indices)))

#     def get_mask(self, search_term, target_indices):
#         """
#         Create a boolean mask for target_indices based on whether each index
#         is in the descendants of the search_term.
#         """
#         all_indices = set(self.get_all_indices_for_search(search_term))
#         return np.array([idx in all_indices for idx in target_indices])

#     def get_all_semantic_masks(self, target_indices):
#         """
#         Create boolean masks for all nodes using the create_mask method.
        
#         Parameters:
#         - target_indices: A list of indices to check against each node's descendants
        
#         Returns:
#         - mask_array: A numpy array of shape (num_nodes, len(target_indices))
#                     where mask_array[i, j] is True if target_indices[j] is in node i's descendants
#         - node_names: List of node names corresponding to rows in the mask_array
#         """
#         import numpy as np

#         # Get list of all synset names
#         node_names = list(self.data.keys())
#         num_nodes = len(node_names)

#         # Initialize list to store masks
#         masks = []

#         # For each node, create a mask using the existing create_mask method
#         for synset_name in node_names:
#             # Get the base name without the synset identifier
#             base_name = synset_name.split('.')[0]

#             # Use the existing create_mask method
#             mask = self.get_mask(base_name, target_indices)
#             masks.append(mask)

#         # Convert list of masks to numpy array
#         mask_array = np.array(masks)

#         return mask_array, node_names

#     def get_all_imagenet_masks(self, target_indices):
#         """
#         Create masks for the 1000 ImageNet classes, using the most specific class name for each index.
        
#         Parameters:
#         - target_indices: A list of indices to check
        
#         Returns:
#         - mask_array: A numpy array of shape (1000, len(target_indices))
#         - imagenet_class_names: List of the 1000 specific class names
#         """
#         import numpy as np
        
#         # For each index, find the most specific class (i.e., the class with the longest path)
#         index_to_specific_class = {}
        
#         for synset_name, data in self.data.items():
#             path_length = len(data['path'].split('.'))
            
#             for idx in data['indices']:
#                 if 0 <= idx < 1000:
#                     # If we haven't seen this index before, or if this class is more specific
#                     if idx not in index_to_specific_class or \
#                     path_length > len(self.data[index_to_specific_class[idx]]['path'].split('.')):
#                         index_to_specific_class[idx] = synset_name
        
#         # Create a list of class names, one for each index
#         class_names = []
#         for idx in range(1000):
#             if idx in index_to_specific_class:
#                 class_names.append(index_to_specific_class[idx])
#             else:
#                 print(f"Warning: No class found for index {idx}")
#                 # Use a placeholder if no class is found
#                 class_names.append(f"unknown_class_{idx}")
        
#         # Create masks for each index
#         masks = []
#         for idx in range(1000):
#             # Simple mask where target index equals current index
#             mask = np.array([i == idx for i in target_indices])
#             masks.append(mask)
        
#         # Convert list of masks to numpy array
#         mask_array = np.array(masks)
        
#         return mask_array, class_names
        
#     def get_normalized_distance(self, concept, from_top=True, use_global_max=False):
#         """
#         Get normalized distance (0-1) for a concept from top or bottom of the hierarchy.
        
#         Parameters:
#             concept (str): Concept name (e.g., 'dog')
#             from_top (bool): If True, distance is measured from root to concept.
#                             If False, distance is measured from concept to leaf.
#             use_global_max (bool): If True, normalize against global max depth of hierarchy.
#                                 If False, normalize within concept's own branch.
        
#         Returns:
#             float or None: Normalized distance in [0, 1] or None if concept not found.
#         """
#         # Find the concept using same method as get_indices
#         matches = self.indices_helper(concept, partial_match=True)

#         if not matches:
#             print(f"[Warning] Concept '{concept}' not found in semantic data.")
#             return None

#         # Choose the best match (you can modify sorting to prefer specific heuristics)
#         concept_name = sorted(matches.keys(), key=lambda k: len(self.data[k]['path']))[0]
#         concept_data = self.data[concept_name]

#         # Get concept distances
#         distance_from_root = concept_data.get('distance_from_root', 0)
#         distance_to_leaves = concept_data.get('distance_to_leaves', 0)


#         # Calculate the normalization factor
#         if use_global_max:
#             max_depth = 0
#             for info in self.data.values():
#                 branch_depth = info.get('distance_from_root', 0) + info.get('distance_to_leaves', 0)
#                 max_depth = max(max_depth, branch_depth)
#             normalizer = max_depth
#         else:
#             normalizer = distance_from_root + distance_to_leaves

#         if normalizer <= 0:
#             return 0.0

#         if from_top:
#             return distance_from_root / normalizer
#         else:
#             return distance_to_leaves / normalizer if not use_global_max else (normalizer - distance_from_root) / normalizer

#     def get_concepts_from_path(self, concept):
#         """
#         Get all concept names from paths containing the specified concept,
#         preserving the original order they appear in each path.
        
#         Parameters:
#         - concept: String representing the starting concept to search for
        
#         Returns:
#         - List of concept names in the order they appear in paths
#         """
#         # Get all descendant nodes for the concept
#         descendants = self.get_descendant_nodes(concept)
        
#         # Track all unique paths to handle duplicates while preserving order
#         all_paths = set()
#         for info in descendants.values():
#             all_paths.add(info['path'])
        
#         # Process each unique path and extract names in order
#         ordered_names = []
#         seen_names = set()  # To track duplicates
        
#         for path in all_paths:
#             # Split the path into components
#             components = path.split(".")
            
#             # Process components in groups of 3 (name, pos, number)
#             i = 0
#             while i < len(components):
#                 # Add the name component if not already seen
#                 if i < len(components) and components[i] not in seen_names:
#                     ordered_names.append(components[i])
#                     seen_names.add(components[i])
                
#                 # Skip to the next name
#                 if i + 2 < len(components) and components[i+1] == 'n':
#                     i += 3  # Standard case: skip name, 'n', and number
#                 else:
#                     i += 1  # Fallback: move forward one component
        
#         return ordered_names

            
# if __name__ == "__main__":
#     sem = SemanticAnalyzer()
#     embed()



