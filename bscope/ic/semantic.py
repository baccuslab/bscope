import os
import json
import tqdm
import numpy as np


def load_imagenet_metadata(synset_path='/mnt/data/ic/synsets.txt'):
    """Load synset names into a dictionary"""
    with open(sysnet_path, "r") as file:
        synset_list = file.read().strip().split("\n")

    synset_map = {}
    for synset in synset_list:
        parts = synset.split(" ")
        synset_id = parts[0]  # Extract ID (e.g., "n01440764")
        synset_name = " ".join(parts[1:])  # Extract readable name
        synset_map[synset_id] = synset_name

    return synset_map


def extract_child(node, mapping):
    """Recursively extract {id: index} from mobilenet.json"""
    if "id" in node and "index" in node:
        mapping[node["id"]] = node["index"]
    if "children" in node:
        for child in node["children"]:
            extract_child(child, mapping)


def load_hierarchy(json_path='/mnt/data/ic/mobilenet.json'):
    """Load MobileNet JSON and extract mapping"""
    with open(json_path, "r") as file:
        hierarchy = json.load(file)

    hierarchy_children = {}
    extract_child(hierarchy, hierarchy_children)
    return hierarchy, hierarchy_children


def get_all_hierarchy_names(node):
    names = set()

    def traverse(node):
        if 'name' in node:
            names.add(node['name'])
        if 'children' in node:
            for child in node['children']:
                traverse(child)

    traverse(node)
    return names


def get_all_children_indices(hierarchy, class_label):
    indices = []

    def search_children(node):
        # If the current node matches the higher-order label, collect indices from its subtree
        if node['name'].startswith(class_label):
            collect_indices(node)

        # If the current node has children, search through them
        if 'children' in node:
            for child in node['children']:
                search_children(child)

    def collect_indices(node):
        # If the node has an index, add it to the list
        if 'index' in node:
            indices.append(node['index'])
        # If the node has children, continue collecting indices
        if 'children' in node:
            for child in node['children']:
                collect_indices(child)

    search_children(hierarchy)
    return indices


def get_mask_for_semantic(hierarchy, hierarchy_name, targets):
    """Generate a binary mask for a given semantic hierarchy"""
    indices = get_all_children_indices(hierarchy, hierarchy_name)
    indices_set = set(indices)
    return np.array([t in indices_set for t in targets])


def get_masks(targets, savefile='/mnt/data/ic/masks.npz'):
    # if savefile exists, load it
    if os.path.exists(savefile):
        print('Loading masks from file')
        masks = np.load(savefile, allow_pickle=True)
        return masks['mask_matrix'], masks['mask_labels']
    else:
        print('Computing masks and saving at file {}'.format(savefile))

        hierarchy, _ = load_hierarchy()

        all_names = get_all_hierarchy_names(hierarchy)
        masks = {}
        for name in tqdm.tqdm(all_names):
            masks[name] = get_mask_for_semantic(hierarchy, name, targets)

        mask_matrix = np.stack(list(masks.values()), axis=1)
        mask_labels = list(masks.keys())

        # Save the masks to a file
        np.savez(savefile, mask_matrix=mask_matrix, mask_labels=mask_labels)
        return mask_matrix, mask_labels


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
