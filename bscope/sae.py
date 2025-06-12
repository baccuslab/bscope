from IPython import embed
from torch.utils.data import TensorDataset
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from overcomplete.sae import BatchTopKSAE, train_sae

def condense_sae(sae, data, device, bs=1024):
    sae.eval()

    dataset = TensorDataset(torch.from_numpy(data).float())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False)
    loadings_agg = []
    codes_agg = []
    with torch.no_grad():

        for batch in tqdm.tqdm(dataloader):
            batch = batch[0]
            _, codes, loadings, mask = sae(batch.float().to(device))
            loadings = loadings.detach().cpu().numpy()
            codes = codes.detach().cpu().numpy()

            loadings_agg.append(loadings)
            codes_agg.append(codes)

        dictionary = sae.dictionary.get_atoms().detach().cpu().numpy()

    loadings = np.concatenate(loadings_agg, axis=0)
    codes = np.concatenate(codes_agg, axis=0)

    binary_loadings = loadings > 0
    summed_loadings = binary_loadings.sum(0)
    dead_modes = summed_loadings == 0.0 
    alive_modes = summed_loadings > 0.0

    num_dead = dead_modes.sum()
    print(f"Number of dead modes: {num_dead}")

    num_alive = alive_modes.sum()
    print(f"Number of alive modes: {num_alive}")

    loadings = loadings[:, alive_modes]
    dictionary = dictionary[alive_modes]
    
    return loadings, dictionary, codes

