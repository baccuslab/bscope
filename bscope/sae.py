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
#


    return loadings, dictionary, dead   
