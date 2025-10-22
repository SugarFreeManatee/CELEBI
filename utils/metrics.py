import json
import torch
from egg.core.callbacks import WandbLogger
from egg.core.interaction import LoggingStrategy, Interaction
import numpy as np
import elsa as ELSA
import re
import os
from scipy.stats import spearmanr
import time
import torch.nn.functional as F  
from torchmetrics.functional.regression import spearman_corrcoef
import wandb
from egg.core.callbacks import WandbLogger, Callback
from typing import Any, Dict
from collections import deque
from torch.utils.data import Subset
import torch
from collections import defaultdict
import math
import sys


def compute_variation_metrics(prob_tensor, rol_sizes, soft_min = False):

    freedom, entanglement = compute_freedom_entanglement(prob_tensor, rol_sizes, soft_min)
    return {
        "synonymy": compute_synonymy(prob_tensor, rol_sizes, soft_min),
        "homonymy": compute_homonymy(prob_tensor, rol_sizes, soft_min),
        "freedom": freedom,
        "entanglement": entanglement,
    }

def compute_probability_tensor(meanings, token_ids, atom_counts, n_positions, n_chars):
    
    """
    Computes the conditional probability tensor:
    P(char_p=j | atom_r=i) for all roles r.
    
    meanings: [N, R] tensor with atom ids (int)
    token_ids: [N, P] tensor with character ids (int)
    n_roles: number of semantic roles (R)
    n_atoms: number of unique atoms per role (A)
    n_positions: number of positions in message (P)
    n_chars: number of characters in vocab (C)
    
    Returns:
        probs: [R, A, P, C] tensor
    """
    n_roles = len(atom_counts)
    n_atoms = max(atom_counts)
    probs = torch.zeros(n_roles, n_atoms, n_positions, n_chars, device=meanings.device)

    for r in range(n_roles):
        for a in range(atom_counts[r]):
            # Find indices where atom `a` is present in role `r`
            mask = (meanings[:, r] == a)
            if mask.sum() == 0:
                continue
            # Select the corresponding token_ids
            selected_tokens = token_ids[mask]  # shape: [?, P]
            for p in range(n_positions):
                char_ids = selected_tokens[:, p]
                hist = torch.bincount(char_ids, minlength=n_chars).float()
                probs[r, a, p, :] = hist / hist.sum()
    
    return probs

def entropy(prob_tensor, dim=-1):
    eps = 1e-8
    log_probs = (prob_tensor + eps).log2()
    return -torch.sum(prob_tensor * log_probs, dim=dim)


def compute_synonymy(prob_tensor, rol_sizes, soft_min):
    role_averages = []
    for role, role_size in enumerate(rol_sizes):
        u_ent = math.log2(prob_tensor.size(-1))
        entropies = entropy(prob_tensor[role,:role_size,:,:]) / u_ent
        if soft_min:
            min_entropies = masked_min(entropies, dim=-1)
        else:
            min_entropies = torch.min(entropies, dim=-1)[0]
        atom_average = torch.mean(min_entropies, dim=-1)
        role_averages.append(atom_average)
    return torch.mean(torch.stack(role_averages)).item()

def masked_min(tensor: torch.Tensor, dim: int = -1, threshold: float = 1e-8):
    """
    Computes the minimum along a dimension, ignoring values below a given threshold.
    
    Parameters:
    ----------
    tensor : torch.Tensor
        The input tensor.
    dim : int
        Dimension to reduce.
    threshold : float
        Values below this are considered invalid and ignored.
    
    Returns:
    -------
    min_values : torch.Tensor
        Minimum values with small values masked out.
    """
    # Replace small values with +inf so they don't affect min
    masked = tensor.clone()
    masked[masked < threshold] = float('inf')
    
    # Take the min, ignoring masked values
    min_values, _ = torch.min(masked, dim=dim)

    # If all values were masked, replace infs with 0
    min_values[min_values == float('inf')] = 0.0
    return min_values

def compute_homonymy(prob_tensor, rol_sizes, soft_min):
    role_averages = []
    for role, role_size in enumerate(rol_sizes):
        probs = prob_tensor[role, :role_size, :, :]  # [A, P, C]

        # Normalize across atoms
        norm = probs.sum(dim=0, keepdim=True) + 1e-8  # [1, P, C]
        p_atom_given_char = probs / norm  # [A, P, C]

        # Entropy over atoms, for each (p, j)
        entropies = entropy(p_atom_given_char, dim=0)  # [P, C]
        u_ent = math.log2(role_size)
        normalized_entropy = entropies / (u_ent + 1e-8)

        # Clamp for numerical stability
        normalized_entropy = torch.clamp(normalized_entropy, 0.0, 1.0)

        # Min entropy over positions per character
        if soft_min:
            min_entropies = masked_min(normalized_entropy, dim=0)  # [C]
        else:
            min_entropies = torch.min(normalized_entropy, dim=0)[0]  # [C]

        char_average = torch.mean(min_entropies)  # scalar
        role_averages.append(char_average)

    return torch.mean(torch.stack(role_averages)).item()


def compute_freedom_entanglement(prob_tensor, rol_sizes, soft_min):
    role_averages = []
    role_comparisons = []
    role_dists = []
    for role, role_size in enumerate(rol_sizes):
        u_ent = math.log2(prob_tensor.size(-1))
        role_ent = entropy(prob_tensor[role, :role_size, :, :], dim=-1) / u_ent # [A, P]
        mean_ent = role_ent.mean(dim=0)  # [P] 
        role_dists.append(mean_ent)
        if soft_min:
            min_pos_entropy = masked_min(mean_ent, dim=0) 
        else:
            min_pos_entropy = torch.min(mean_ent)
        role_averages.append(min_pos_entropy)
    
    for i in range(len(role_dists)-1):
        for j in range(i+1, len(role_dists)):
            diff = torch.max(torch.abs(role_dists[i] - role_dists[j]))
            max_ = torch.max(torch.stack([role_dists[i], role_dists[j]]))
            role_comparisons.append(diff / (max_ + 1e-8))
    return torch.mean(torch.stack(role_averages)).item(), 1-torch.mean(torch.stack(role_comparisons)).item()    

def get_seen_factors(latents):
    seen_values = []
    for dim in range(latents.shape[1]):
        seen = set(latents[:, dim].tolist())
        seen_values.append(seen)
    return seen_values

def filter_test_by_seen_factors(test_latents, seen_values):
    mask = []
    for latent in test_latents:
        keep = all(latent[dim].item() in seen_values[dim] for dim in range(len(latent)))
        mask.append(keep)
    return torch.tensor(mask)

def subsample_train_and_filter_test(train_set, test_set, n_train, seed=42, verbose=True):
    """
    Subsamples n_train examples from the train_set, and filters the test_set
    to only include examples whose factor values are all present in the train subset.

    Args:
        train_set: full training dataset (must return (image, latents))
        test_set: full test dataset (same format)
        n_train: number of training examples to keep
        seed: random seed for reproducibility
        verbose: whether to print stats

    Returns:
        train_subset: Subset(train_set, train_indices)
        test_subset:  Subset(test_set, test_indices)
    """
    rng = torch.Generator().manual_seed(seed)
    train_indices = torch.randperm(len(train_set), generator=rng)[:n_train]
    train_subset = Subset(train_set, train_indices)

    train_latents = torch.stack([train_set[i][1] for i in train_indices])
    seen_values = get_seen_factors(train_latents)

    test_latents = torch.stack([test_set[i][1] for i in range(len(test_set))])
    test_mask = filter_test_by_seen_factors(test_latents, seen_values)
    test_indices = test_mask.nonzero(as_tuple=True)[0]
    test_subset = Subset(test_set, test_indices)

    if verbose:
        print(f"Selected {len(train_indices)} training examples.")
        print(f"Filtered test set from {len(test_set)} → {len(test_indices)} examples "
              f"({100.0 * len(test_indices) / len(test_set):.1f}%) "
              f"with factors seen in train.")

    return train_subset, test_subset

def reset_module_weights(module):
    for layer in module.modules():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

