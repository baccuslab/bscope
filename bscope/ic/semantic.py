import os
from IPython import embed
import json
import tqdm
import numpy as np
import json
import requests





class SynsetAnalyzer:

    def __init__(self, json_file_path='/mnt/data/ic/node_data_mapping.json'):
        self.data = self.load_data(json_file_path)

    def load_data(self, path):
        """Load synset data from JSON file."""
        with open(path, 'r') as f:
            return json.load(f)

    def get_indices_for_name(self, name, partial_match=False):
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

    def print_hierarchy_summary(self, name, partial_match=False):
        """
        Print matching synsets and also show all indices from all descendants.
        Returns a list of all unique indices from all descendants.
        """
        # First get direct matches
        matches = self.get_indices_for_name(name, partial_match)

        if not matches:
            print(f"No synsets found with name '{name}'.")
            return []

        print(f"Found {len(matches)} matching term for '{name}':")

        # Print info for direct matches
        for synset_name, info in matches.items():
            print(f"\nSynset: {synset_name}")
            print(f"Definition: {info['definition']}")
            print(f"Path: {info['path']}")
            print(f"Number of indices: {len(info['indices'])}")
            if info['indices']:
                preview = ', '.join(map(str, info['indices'][:10]))
                print(f"Indices: {preview}")
                if len(info['indices']) > 10:
                    print(f"...and {len(info['indices']) - 10} more")
            else:
                print("No indices associated with this synset.")

        # Now get ALL descendants and their indices
        all_descendants = self.get_descendant_nodes(name)

        # Collect all indices from all descendants
        all_indices = set()
        for desc_name, desc_info in all_descendants.items():
            if 'indices' in desc_info and desc_info['indices']:
                all_indices.update(desc_info['indices'])

        # Sort the indices
        all_indices = sorted(list(all_indices))

        # Print info about all descendants and their indices
        print(f"\nTotal descendants : {len(all_descendants)}")
        print(f"Total unique indices from all descendants: {len(all_indices)}")
        if all_indices:
            preview = ', '.join(map(str, all_indices[:20]))
            print(f"All indices: {preview}")
            if len(all_indices) > 20:
                print(f"...and {len(all_indices) - 20} more")

        # Return all unique indices for use in mask function
        return all_indices

    def get_parent_nodes(self, name, partial_match=False):
        """Find parent nodes for the given name."""
        matches = self.get_indices_for_name(name, partial_match)
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

    def print_parent_nodes(self, name, partial_match=False):
        """Print parent nodes for the given name."""
        matches = self.get_indices_for_name(name, partial_match)
        for synset in matches:
            print(f"  {synset}")
        print(f"\nSearching for nodes that have '{name}' in their ancestry:")
        parents = self.get_parent_nodes(name, partial_match)
        if not parents:
            print(f"No parent nodes found for synsets with name '{name}'.")
        else:
            print(f"\nFound {len(parents)} parent nodes:")
            for parent_name in parents:
                print(f"  {parent_name}")

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

    def print_descendant_nodes(self, term):
        """Print all nodes that have the term in their ancestry."""
        print(f"\nSearching for nodes that have '{term}' in their ancestry:")
        descendants = self.get_descendant_nodes(term)
        if not descendants:
            print(f"No nodes found with '{term}' in their ancestry.")
        else:
            print(f"\nFound {len(descendants)} descendant nodes:")
            for synset_name in descendants:
                print(f"  {synset_name}")

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

    def create_mask(self, search_term, target_indices):
        """
        Create a boolean mask for target_indices based on whether each index
        is in the descendants of the search_term.
        """
        all_indices = set(self.get_all_indices_for_search(search_term))
        return np.array([idx in all_indices for idx in target_indices])

    def print_all_indices_for_term(self, term):
        """
        Print all indices associated with a term and its descendants.
        """
        indices = self.get_all_indices_for_search(term)
        descendants = self.get_descendant_nodes(term)

        print(f"Results for '{term}':")
        print(f"- Descendants: {len(descendants)} synsets")
        print(f"- Total indices found: {len(indices)}")
        print(f"- Indices: {indices}")

        # Print each descendant and its indices
        for synset_name, info in descendants.items():
            indices = info.get('indices', [])
            if indices:
                print(f"\n{synset_name}: {indices}")

    def get_masks_for_all_nodes(self, target_indices):
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
            mask = self.create_mask(base_name, target_indices)
            masks.append(mask)

        # Convert list of masks to numpy array
        mask_array = np.array(masks)

        return mask_array, node_names

if __name__ == "__main__":
    # Example usage
    analyzer = SynsetAnalyzer()
    embed()
