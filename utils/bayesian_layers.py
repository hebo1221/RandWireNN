"""
Bayesian layers for uncertainty quantification in RandWireNN.

Implements two approaches:
1. MC Dropout: Simple and effective (Gal & Ghahramani, 2016)
2. Variational Inference: Full Bayesian treatment (Blundell et al., 2015)

Based on:
- "Bayesian Randomly Wired Neural Network with Variational Inference" (2020)
- "Dropout as a Bayesian Approximation" (Gal & Ghahramani, 2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class MCDropout(nn.Module):
    """
    Monte Carlo Dropout layer.
    Enables dropout during inference for uncertainty estimation.
    """

    def __init__(self, p: float = 0.5):
        super(MCDropout, self).__init__()
        self.p = p

    def forward(self, x):
        # Always apply dropout (even during eval for MC sampling)
        return F.dropout(x, p=self.p, training=True)


class BayesianConv2d(nn.Module):
    """
    Bayesian Convolutional layer using variational inference.

    Each weight has a distribution instead of a point estimate.
    Uses reparameterization trick for backpropagation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, groups=1, prior_std=1.0):
        super(BayesianConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.prior_std = prior_std

        # Weight mean and log variance (rho)
        self.weight_mu = nn.Parameter(torch.Tensor(
            out_channels, in_channels // groups, *self.kernel_size))
        self.weight_rho = nn.Parameter(torch.Tensor(
            out_channels, in_channels // groups, *self.kernel_size))

        # Bias mean and log variance
        self.bias_mu = nn.Parameter(torch.Tensor(out_channels))
        self.bias_rho = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize using Kaiming initialization
        nn.init.kaiming_normal_(self.weight_mu, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.weight_rho, -3)  # Small initial variance
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_rho, -3)

    def forward(self, x, sample=True):
        if sample:
            # Sample weights from distribution
            weight_std = torch.log1p(torch.exp(self.weight_rho))
            weight = self.weight_mu + weight_std * torch.randn_like(self.weight_mu)

            bias_std = torch.log1p(torch.exp(self.bias_rho))
            bias = self.bias_mu + bias_std * torch.randn_like(self.bias_mu)
        else:
            # Use mean values (MAP estimate)
            weight = self.weight_mu
            bias = self.bias_mu

        return F.conv2d(x, weight, bias, self.stride, self.padding, groups=self.groups)

    def kl_divergence(self):
        """
        Compute KL divergence between posterior and prior.
        KL(q(w|θ) || p(w))
        """
        weight_std = torch.log1p(torch.exp(self.weight_rho))
        bias_std = torch.log1p(torch.exp(self.bias_rho))

        # KL for weights
        kl_weight = (
            torch.log(self.prior_std / weight_std) +
            (weight_std**2 + self.weight_mu**2) / (2 * self.prior_std**2) -
            0.5
        ).sum()

        # KL for bias
        kl_bias = (
            torch.log(self.prior_std / bias_std) +
            (bias_std**2 + self.bias_mu**2) / (2 * self.prior_std**2) -
            0.5
        ).sum()

        return kl_weight + kl_bias


class BayesianLinear(nn.Module):
    """
    Bayesian Linear layer using variational inference.
    """

    def __init__(self, in_features, out_features, prior_std=1.0):
        super(BayesianLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std

        # Weight mean and log variance
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))

        # Bias mean and log variance
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight_mu)
        nn.init.constant_(self.weight_rho, -3)
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_rho, -3)

    def forward(self, x, sample=True):
        if sample:
            weight_std = torch.log1p(torch.exp(self.weight_rho))
            weight = self.weight_mu + weight_std * torch.randn_like(self.weight_mu)

            bias_std = torch.log1p(torch.exp(self.bias_rho))
            bias = self.bias_mu + bias_std * torch.randn_like(self.bias_mu)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def kl_divergence(self):
        weight_std = torch.log1p(torch.exp(self.weight_rho))
        bias_std = torch.log1p(torch.exp(self.bias_rho))

        kl_weight = (
            torch.log(self.prior_std / weight_std) +
            (weight_std**2 + self.weight_mu**2) / (2 * self.prior_std**2) -
            0.5
        ).sum()

        kl_bias = (
            torch.log(self.prior_std / bias_std) +
            (bias_std**2 + self.bias_mu**2) / (2 * self.prior_std**2) -
            0.5
        ).sum()

        return kl_weight + kl_bias


class BayesianBatchNorm2d(nn.Module):
    """
    Bayesian Batch Normalization.
    Combines standard BN with uncertainty in scale/shift parameters.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(BayesianBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine)

        if affine:
            # Add variational parameters for scale and shift
            self.weight_rho = nn.Parameter(torch.zeros(num_features))
            self.bias_rho = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, sample=True):
        # Standard batch norm
        x = self.bn(x)

        if self.bn.affine and sample:
            # Add variational noise to scale and bias
            weight_std = torch.log1p(torch.exp(self.weight_rho))
            bias_std = torch.log1p(torch.exp(self.bias_rho))

            weight_noise = weight_std * torch.randn_like(self.bn.weight)
            bias_noise = bias_std * torch.randn_like(self.bn.bias)

            # Apply noisy transformation
            x = x * (1 + weight_noise).view(1, -1, 1, 1)
            x = x + bias_noise.view(1, -1, 1, 1)

        return x


def convert_to_bayesian(model, dropout_p=0.1, use_variational=False):
    """
    Convert a standard model to Bayesian by replacing layers.

    Args:
        model: PyTorch model to convert
        dropout_p: Dropout probability for MC Dropout
        use_variational: If True, use variational layers; else use MC Dropout

    Returns:
        Modified model with Bayesian layers
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Dropout):
            # Replace with MC Dropout
            setattr(model, name, MCDropout(p=dropout_p))
        elif isinstance(module, nn.Conv2d) and use_variational:
            # Replace with Bayesian Conv
            new_layer = BayesianConv2d(
                module.in_channels, module.out_channels,
                module.kernel_size, module.stride,
                module.padding, module.groups
            )
            setattr(model, name, new_layer)
        elif isinstance(module, nn.Linear) and use_variational:
            # Replace with Bayesian Linear
            new_layer = BayesianLinear(module.in_features, module.out_features)
            setattr(model, name, new_layer)
        else:
            # Recursively convert child modules
            convert_to_bayesian(module, dropout_p, use_variational)

    return model


def compute_kl_loss(model):
    """
    Compute total KL divergence for all Bayesian layers in model.

    Args:
        model: Model with Bayesian layers

    Returns:
        Total KL divergence
    """
    kl_loss = 0
    for module in model.modules():
        if isinstance(module, (BayesianConv2d, BayesianLinear)):
            kl_loss += module.kl_divergence()
    return kl_loss
