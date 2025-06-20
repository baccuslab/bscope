from IPython import embed
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os


class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma
    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.sigma
            return x + noise
        return x

def r2_score(x, x_hat, eps=1e-6):
    assert x.shape == x_hat.shape, "Input tensors must have the same shape"
    assert len(x.shape) == 2, "Input tensors must be 2D"

    ss_res = torch.mean((x - x_hat) ** 2)
    ss_tot = torch.mean((x - x.mean()) ** 2)

    r2 = 1 - (ss_res / (ss_tot + eps))

    return torch.mean(r2)


def coherence_regularization(W, normalize=True):
    """
    Compute coherence regularization penalty for matrix W.
    
    Args:
        W: torch.Tensor of shape (d, k) where d is feature dimension, k is number of columns
        normalize: bool, whether to normalize columns before computing coherence
    
    Returns:
        coherence_penalty: scalar tensor
    """
    W = W.T
    if normalize:
        # Normalize columns to unit norm
        W_norm = F.normalize(W, p=2, dim=0)
    else:
        W_norm = W
    
    # Compute gram matrix (column correlations)
    gram = torch.mm(W_norm.t(), W_norm)  # Shape: (k, k)
    
    # Zero out diagonal (self-correlations)
    mask = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
    gram_off_diag = gram * (1 - mask)
    
    # Sum of absolute off-diagonal correlations
    coherence_penalty = torch.sum(torch.abs(gram_off_diag))
    return coherence_penalty/ W_norm.size(1)  # Normalize by number of columns


class Dictionary(nn.Module):
    def __init__(self, num_atoms, atom_dim):
        super(Dictionary, self).__init__()
        self.num_atoms = num_atoms
        self.atom_dim = atom_dim
        self.atoms = nn.Parameter(torch.rand(num_atoms, atom_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.atoms)  # Initialize atoms with Xavier uniform distribution
        self.sign_constraint = None# Options: 'positive', 'negative', 'none'N
        # nn.init.orthogonal_(self.atoms)  # Initialize atoms with orthogonal vectors
        self.relu = nn.ReLU()


    def forward(self, x):
        atoms = self.get_atoms()

        if self.sign_constraint == 'positive':
            atoms = self.relu(atoms)  # Apply ReLU to atoms

        return torch.matmul(x, atoms)

    def get_atoms(self):
        atoms = self.atoms
        return atoms

class Encoder(nn.Module):
    def __init__(self, data_dim, num_atoms, mlp_hidden_dim=512):
        super(Encoder, self).__init__()
        self.data_dim = data_dim
        self.num_atoms = num_atoms
        self.mlp_hidden_dim = mlp_hidden_dim
        self.layers = nn.ModuleDict()
        self.layers['layernorm1'] = nn.LayerNorm(data_dim, elementwise_affine=True)
        self.layers['layer1'] = nn.Linear(data_dim, self.mlp_hidden_dim, bias=True)
        self.layers['relu1'] = nn.ReLU()
        nn.init.xavier_uniform_(self.layers['layer1'].weight)  # Initialize weights with Xavier uniform distribution 
        self.layers['layernorm2'] = nn.LayerNorm(self.mlp_hidden_dim, elementwise_affine=True)
        self.layers['layer2'] = nn.Linear(self.mlp_hidden_dim, num_atoms, bias=False)
        nn.init.xavier_uniform_(self.layers['layer2'].weight)  # Initialize weights with Xavier uniform distribution
        self.layers['sigmoid'] = nn.Sigmoid()


    def forward(self, x):
        for layer in self.layers.values():
            x = layer(x)
        return x



class SigSigSAE(nn.Module):
    def __init__(self, data_dim, num_atoms, a, b, mlp_hidden_dim=512, sigma=0.05, sign_constraint=None):
        super(SigSigSAE, self).__init__()
        self.encoder = Encoder(data_dim, num_atoms,mlp_hidden_dim)
        self.dictionary = Dictionary(num_atoms, data_dim)

        self.dictionary.sign_constraint = sign_constraint  # Set sign constraint for dictionary atoms
        
        self.a = a
        self.b = b

        self.noise = GaussianNoise(sigma=sigma)  # Add Gaussian noise with sigma=0.1

        # x = np.linspace(0, 1, 1000)
        # x = torch.tensor(x, dtype=torch.float32)

        # y = self.sigmoid(x, a=self.a, b=self.b)  # Example parameters for steep sigmoid
        # y = y.detach().numpy()

        # plt.plot(x, y)
        # plt.show()


    def sigmoid(self, x, a, b):
        """
        Sigmoid function with parameters a and b.

        Args:
            x (torch.Tensor): Input tensor.
            a (float): Steepness of the sigmoid curve.
            b (float): Horizontal shift of the sigmoid curve.

        An example of a very steep sigmoid function:

        """
        s = torch.clip(x, min=1e-8, max=1 - 1e-8)  # Avoid log(0) issues
        s = 1 / (1 + torch.exp(-a * (x - b)))
        s = torch.clamp(s, min=1e-8, max=1 - 1e-8)  # Avoid log(0) issues
        return s 
    def forward(self, x):
        # If training
        # if self.training:
        if self.training:
            codes = self.encoder(x)
            z = self.sigmoid(codes, a=self.a, b=self.b)
            mask = torch.ones_like(codes).float().detach()  # Use ones to keep all codes 
            reconstructed = self.dictionary(z)
            return reconstructed, codes, z, mask
        else:
            codes = self.encoder(x)
            z= self.sigmoid(codes, a=self.a, b=self.b)

            mask = (codes >= self.b).float().detach()
            mask= z* mask

            reconstructed = self.dictionary(mask)

            return reconstructed, codes, z, mask 


class SAE(nn.Module):
    def __init__(self, data_dim, num_atoms, threshold = 0.95, mlp_hidden_dim=512):
        super(SAE, self).__init__()
        self.encoder = Encoder(data_dim, num_atoms,mlp_hidden_dim)
        self.dictionary = Dictionary(num_atoms, data_dim)

        self.threshold = threshold
    
    def forward(self, x):
        codes = self.encoder(x)

        mask = (codes >= self.threshold).float().detach()
        z = codes * mask

        reconstructed = self.dictionary(z)
        return reconstructed, codes, z, mask



def load_sigsig_sae(path, data, device, bs=1024, eval_mode=True):
    sae = torch.load(path, map_location=device, weights_only=False)

    if eval_mode:
        sae.eval()
    else:
        sae.train()

    dataset = TensorDataset(torch.from_numpy(data).float())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False)

    loadings_agg = []
    codes_agg = []
    reconstructed_agg = []
    data_agg = []
    with torch.no_grad():

        for batch in tqdm.tqdm(dataloader):
            batch = batch[0]
            data_agg.append(batch.detach().cpu().numpy())
            reco, codes, loadings, mask = sae(batch.float().to(device))
            loadings = mask.detach().cpu().numpy()
            codes = codes.detach().cpu().numpy()
            reconstructed_agg.append(reco.detach().cpu().numpy())
            loadings_agg.append(loadings)
            codes_agg.append(codes)

        dictionary = sae.dictionary.get_atoms().detach().cpu().numpy()
    reconstructed_agg = np.concatenate(reconstructed_agg, axis=0)
    data_agg = np.concatenate(data_agg, axis=0)

    r2 = r2_score(torch.from_numpy(data_agg).float(), torch.from_numpy(reconstructed_agg).float()).item()
    print('R2 score: ',r2)
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
    
    return sae, loadings, dictionary, r2

def load_sae(path, data, device, bs=1024, eval_mode=True):
    sae = torch.load(path, map_location=device, weights_only=False)

    # Check if the model is a SigSigSAE or SAE
    if isinstance(sae, SigSigSAE):
        print('Found SigSigSAE model')
        return load_sigsig_sae(path, data, device, bs, eval_mode)
    else:
        pass


    if eval_mode:
        sae.eval()
    else:
        sae.train()

    dataset = TensorDataset(torch.from_numpy(data).float())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False)

    loadings_agg = []
    codes_agg = []
    reconstructed_agg = []
    data_agg = []
    with torch.no_grad():

        for batch in tqdm.tqdm(dataloader):
            batch = batch[0]
            data_agg.append(batch.detach().cpu().numpy())
            reco, codes, loadings, mask = sae(batch.float().to(device))
            loadings = loadings.detach().cpu().numpy()
            codes = codes.detach().cpu().numpy()
            reconstructed_agg.append(reco.detach().cpu().numpy())
            loadings_agg.append(loadings)
            codes_agg.append(codes)

        dictionary = sae.dictionary.get_atoms().detach().cpu().numpy()
    reconstructed_agg = np.concatenate(reconstructed_agg, axis=0)
    data_agg = np.concatenate(data_agg, axis=0)

    r2 = r2_score(torch.from_numpy(data_agg).float(), torch.from_numpy(reconstructed_agg).float()).item()
    print('R2 score: ',r2)
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
    
    return sae, loadings, dictionary, r2

