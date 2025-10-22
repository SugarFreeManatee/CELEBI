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

def compute_topsim_zakajd(
        # https://discuss.pytorch.org/t/spearmans-correlation/91931/5
        meanings: torch.Tensor,
        messages: torch.Tensor,
        meaning_distance_fn="hamming",
        message_distance_fn="hamming",
    ) -> float:

        if meaning_distance_fn == "hamming":
            meaning_dist = F.pdist(meanings, p=0) / meanings.size(-1)
        if message_distance_fn == "euclidean":
            message_dist = F.pdist(messages, p=2)
        if message_distance_fn == "hamming":
            message_dist = F.pdist(messages, p=0) / messages.size(-1)
        topsim = spearman_corrcoef(meaning_dist, message_dist)

        return topsim.mean(), message_dist.mean()


def pdm(student_message, straight_through=False):  # student_message -> (B, C, V)
    # Compute dot products pairwise between all vectors
    dot_product = torch.einsum('bcv,dcv->bd', student_message, student_message)  # (B, B)

    if not straight_through:  # Norms are not guaranteed to be 1
        norms = student_message.norm(dim=-1)  # Compute norms along the last axis (V)
        norm_product = torch.einsum('bc,dc->bd', norms, norms)  # Pairwise norm products
        cosine_sim = dot_product / norm_product.clamp(min=1e-8)  # Avoid division by zero
    else:
        cosine_sim = dot_product  # Assumes normalized vectors

    # Set diagonal to zero
    cosine_sim.fill_diagonal_(0)

    return cosine_sim

def koleo_loss(X, straight_through = False, eps=1e-6):
    
    dot_product = torch.einsum('bcv,dcv->bd', X, X)  # (B, B)

    if not straight_through:  # Norms are not guaranteed to be 1
        norms = X.norm(dim=-1)  # Compute norms along the last axis (V)
        norm_product = torch.einsum('bc,dc->bd', norms, norms)  # Pairwise norm products
        cosine_sim = dot_product / norm_product.clamp(min=1e-8)  # Avoid division by zero
    else:
        cosine_sim = dot_product  # Assumes normalized vectors
    
    cosine_sim.fill_diagonal_(0)
    cosine_dist = 1 - cosine_sim
    
    nn_dists, _ = cosine_dist.min(dim=1)  # nearest neighbor
    return -torch.log(nn_dists + eps)

def kl_divergence_loss(receiver_output, unrolled_X):
    """
    Compute the KL divergence between two sequences of Gaussian distributions.

    Args:
        mu1: Tensor of shape (batch_size, seq_length, latent_dim), mean of the first Gaussian sequence.
        logvar1: Tensor of shape (batch_size, seq_length, latent_dim), log variance of the first Gaussian sequence.
        mu2: Tensor of shape (batch_size, seq_length, latent_dim), mean of the second Gaussian sequence.
        logvar2: Tensor of shape (batch_size, seq_length, latent_dim), log variance of the second Gaussian sequence.

    Returns:
        kl_div: Tensor of shape (batch_size, seq_length), KL divergence for each sequence element.
    """
    dim = receiver_output.size(-1)//2
    mu1, logvar1 = receiver_output[:,:,:dim], receiver_output[:,:,dim:]
    mu2, logvar2 = unrolled_X[:,:,:dim], unrolled_X[:,:,dim:]

    var1 = logvar1.exp()  # Variance of the first Gaussian
    var2 = logvar2.exp()  # Variance of the second Gaussian

    # Apply the KL divergence formula element-wise
    kl_div = 0.5 * (
        torch.log(var2 / var1) - 1 + var1 / var2 + ((mu2 - mu1) ** 2) / var2
    ).mean(dim=-1)  # Sum over the latent dimension (last dimension)

    return kl_div


def wasserstein_loss(receiver_output, unrolled_X):
    """
    Computes the squared 2-Wasserstein distance between two batches of Gaussian distributions with diagonal covariance.
    
    Args:
        mu1 (torch.Tensor): Means of the first batch, shape (N, D).
        logvar1 (torch.Tensor): Log-variances of the first batch, shape (N, D).
        mu2 (torch.Tensor): Means of the second batch, shape (N, D).
        logvar2 (torch.Tensor): Log-variances of the second batch, shape (N, D).
    
    Returns:
        torch.Tensor: The squared 2-Wasserstein distance for each batch, shape (N,).
    """
    dim = receiver_output.size(-1)//2
    mu1, logvar1 = receiver_output[:,:,:dim], receiver_output[:,:,dim:]
    mu2, logvar2 = unrolled_X[:,:,:dim], unrolled_X[:,:,dim:]

    # Mean term: ||mu1 - mu2||^2
    mean_diff = torch.mean((mu1 - mu2)**2, dim=-1)
    
    # Variance term
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)
    
    # Compute trace term for diagonal matrices
    trace_term = torch.mean(var1 + var2 - 2 * torch.sqrt(var1 * var2), dim = -1)
    
    # Wasserstein distance
    wasserstein_dist_sq = mean_diff + trace_term
    return wasserstein_dist_sq

def MSE_loss(receiver_output, unrolled_X):
    return F.mse_loss(receiver_output, unrolled_X, reduction = 'none').mean(-1)

def z_MSE_loss(receiver_output, unrolled_X):
    dim = receiver_output.size(-1)//2
    mu1, logvar1 = receiver_output[:,:,:dim], receiver_output[:,:,dim:]
    mu2, logvar2 = unrolled_X[:,:,:dim], unrolled_X[:,:,dim:]
    z1, z2 = reparameterize(mu1, logvar1), reparameterize(mu2, logvar2)
    return MSE_loss(z1, z2)
    
def reparameterize(mu, logvar):
    var = torch.exp(0.5 * logvar)  # Convert log variance to standard deviation
    epsilon = torch.randn_like(var)  # Sample standard normal noise
    return mu + var * epsilon 