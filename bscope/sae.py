from IPython import embed
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from overcomplete.sae import BatchTopKSAE, train_sae

def load_sae(path, data, device):

    sae = torch.load(path, weights_only=False, map_location=device) 

    sae = sae.eval()
#
    with torch.no_grad():
        pre_loadings, loadings = sae.encode(
            torch.from_numpy(data).float().to(device))
#
    loadings = loadings.cpu().numpy()
#
    z = loadings > 0
    dead = z.sum(0) == 0

    loadings = loadings[:, ~dead]
    dictionary = sae.get_dictionary().detach().cpu().numpy()[~dead, :]


    return loadings, dictionary, dead   

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

