from IPython import embed
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from overcomplete.sae import BatchTopKSAE, train_sae

def load_sae(path, data, device, threshold=25, eval_mode=True, reconstruct=False):

    sae = torch.load(path, weights_only=False, map_location=device) 
    if eval_mode:
        sae = sae.eval()
    else:
        sae = sae.train()
#
    with torch.no_grad():
        if reconstruct:
            print('reconstructing')
            pre_loadings, loadings, recodata  = sae(
            torch.from_numpy(data).float().to(device))
        else:
            pre_loadings, loadings= sae.encode(
            torch.from_numpy(data).float().to(device))
#
    loadings = loadings.cpu().numpy()
    dictionary = sae.get_dictionary().detach().cpu().numpy()
    
    binary_loadings = loadings > 0
    summed_loadings = binary_loadings.sum(0)
    dead_modes = summed_loadings < threshold
    alive_modes = summed_loadings >= threshold

    num_dead = dead_modes.sum()
    print(f"Number of dead modes: {num_dead}")

    num_alive = alive_modes.sum()
    print(f"Number of alive modes: {num_alive}")

    loadings = loadings[:, alive_modes]
    dictionary = dictionary[alive_modes]
    
    if reconstruct:
        return recodata, loadings, dictionary, (num_alive, num_dead)
    else:
        return loadings, dictionary, (num_alive, num_dead)

# class SAE(nn.Module):
#     def __init__(self,n_total_modes, device='cpu'):
#         super().__init__()
#         self.nb_concepts = nb_concepts
#         self.device = device

#     def encode(self, x):
#         pre_codes, codes = self.encoder(x)
#         return pre_codes, codes

#     def decode(self, z):
#         return self.dictionary(z)

#     def get_dictionary(self):
#         return self.dictionary.get_dictionary()

#     def forward(self, x):
#         pre_codes, codes = self.encode(x)
#         x_reconstructed = self.decode(codes)
#         return pre_codes, codes, x_reconstructed

# class Dictionary(nn.Module):
#     def __init__(self, in_dimensions, nb_concepts, device='cpu', initializer=None, use_multiplier=True, normalization=None):
#         super().__init__()
#         self.in_dimensions = in_dimensions
#         self.nb_concepts = nb_concepts
#         self.device = device

