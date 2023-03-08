# MIT License

# Copyright (c) 2021 Pashupati Hegde.
# Copyright (c) 2023 Magnus Ross.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import functorch
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import init

from hgp.misc.constraint_utils import invsoftplus, softplus

from ..misc.ham_utils import build_J

prior_weights = Normal(0.0, 1.0)


def sample_normal(shape, seed=None):
    rng = np.random.RandomState() if seed is None else np.random.RandomState(seed)
    return torch.tensor(rng.normal(size=shape).astype(np.float32))


class RBF(torch.nn.Module):
    """
    Implements squared exponential kernel with kernel computation and weights and frquency sampling for Fourier features
    """

    def __init__(self, D_in, D_out=None, dimwise=False, init_ls=2.0, init_var=0.5):
        """
        @param D_in: Number of input dimensions
        @param D_out: Number of output dimensions
        @param dimwise: If True, different kernel parameters are given to output dimensions
        """
        super(RBF, self).__init__()
        self.D_in = D_in
        self.D_out = D_in if D_out is None else D_out
        self.dimwise = dimwise
        lengthscales_shape = (self.D_out, self.D_in) if dimwise else (self.D_in,)
        variance_shape = (self.D_out,) if dimwise else (1,)

        self.unconstrained_lengthscales = nn.Parameter(
            torch.ones(size=lengthscales_shape), requires_grad=True
        )
        self.unconstrained_variance = nn.Parameter(
            torch.ones(size=variance_shape), requires_grad=True
        )
        self._initialize(init_ls, init_var)

    def _initialize(self, init_ls, init_var):
        init.constant_(
            self.unconstrained_lengthscales, invsoftplus(torch.tensor(init_ls)).item()
        )
        init.constant_(
            self.unconstrained_variance, invsoftplus(torch.tensor(init_var)).item()
        )

    @property
    def lengthscales(self):
        return softplus(self.unconstrained_lengthscales)

    @lengthscales.setter
    def lengthscales(self, value):
        self.unconstrained_lengthscales = nn.Parameter(
            invsoftplus(value), requires_grad=True
        )

    @property
    def variance(self):
        return softplus(self.unconstrained_variance)

    @variance.setter
    def variance(self, value):
        self.unconstrained_variance = nn.Parameter(
            invsoftplus(value), requires_grad=True
        )

    def square_dist_dimwise(self, X, X2=None):
        """
        Compues squared euclidean distance (scaled) for dimwise kernel setting
        @param X: Input 1 (N,D_in)
        @param X2: Input 2 (M,D_in)
        @return: Tensor (D_out, N,M)
        """
        X = X.unsqueeze(0) / self.lengthscales.unsqueeze(1)  # (D_out,N,D_in)
        Xs = torch.sum(torch.pow(X, 2), dim=2)  # (D_out,N)
        if X2 is None:
            return (
                -2 * torch.einsum("dnk, dmk -> dnm", X, X)
                + Xs.unsqueeze(-1)
                + Xs.unsqueeze(1)
            )  # (D_out,N,N)
        else:
            X2 = X2.unsqueeze(0) / self.lengthscales.unsqueeze(1)  # (D_out,M,D_in)
            X2s = torch.sum(torch.pow(X2, 2), dim=2)  # (D_out,N)
            return (
                -2 * torch.einsum("dnk, dmk -> dnm", X, X2)
                + Xs.unsqueeze(-1)
                + X2s.unsqueeze(1)
            )  # (D_out,N,M)

    def square_dist(self, X, X2=None):
        """
        Compues squared euclidean distance (scaled) for non dimwise kernel setting
        @param X: Input 1 (N,D_in)
        @param X2: Input 2 (M,D_in)
        @return: Tensor (N,M)
        """
        X = X / self.lengthscales  # (N,D_in)
        Xs = torch.sum(torch.pow(X, 2), dim=1)  # (N,)
        if X2 is None:
            return (
                -2 * torch.matmul(X, X.t())
                + torch.reshape(Xs, (-1, 1))
                + torch.reshape(Xs, (1, -1))
            )  # (N,1)
        else:
            X2 = X2 / self.lengthscales  # (M,D_in)
            X2s = torch.sum(torch.pow(X2, 2), dim=1)  # (M,)
            return (
                -2 * torch.matmul(X, X2.t())
                + torch.reshape(Xs, (-1, 1))
                + torch.reshape(X2s, (1, -1))
            )  # (N,M)

    def K(self, X, X2=None):
        """
        Computes K(\X, \X_2)
        @param X: Input 1 (N,D_in)
        @param X2:  Input 2 (M,D_in)
        @return: Tensor (D,N,M) if dimwise else (N,M)
        """
        if self.dimwise:
            sq_dist = torch.exp(-0.5 * self.square_dist_dimwise(X, X2))  # (D_out,N,M)
            return self.variance[:, None, None] * sq_dist  # (D_out,N,M)
        else:
            sq_dist = torch.exp(-0.5 * self.square_dist(X, X2) / 2)  # (N,M)
            return self.variance * sq_dist  # (N,M)

    def sample_freq(self, S, seed=None):
        """
        Computes random samples from the spectral density for Sqaured exponential kernel
        @param S: Number of features
        @param seed: random seed
        @return: Tensor a random sample from standard Normal (D_in, S, D_out) if dimwise else (D_in, S)
        """
        omega_shape = (self.D_in, S, self.D_out) if self.dimwise else (self.D_in, S)
        omega = sample_normal(omega_shape, seed)  # (D_in, S, D_out) or (D_in, S)
        lengthscales = (
            self.lengthscales.T.unsqueeze(1)
            if self.dimwise
            else self.lengthscales.unsqueeze(1)
        )  # (D_in,1,D_out) or (D_in,1)
        return omega / lengthscales  # (D_in, S, D_out) or (D_in, S)


class DerivativeRBF(RBF):
    """
    Implements squared exponential kernel with kernel computation and weights and frquency sampling for Fourier features.
    Additionally implements gradients and hessians of kernels, only applies for single output.
    """

    def __init__(self, D_in, init_ls=2.0, init_var=0.5):

        assert D_in % 2 == 0, "D_in must be even."

        super(DerivativeRBF, self).__init__(
            D_in, D_out=1, dimwise=False, init_ls=init_ls, init_var=init_var
        )
        self.J = build_J(D_in)

    def single_k(self, xi, yi):
        """Kernel at a single point"""
        xi = xi / self.lengthscales
        yi = yi / self.lengthscales
        return self.variance[0] * torch.exp(-0.5 * torch.sum((xi - yi) ** 2 / 2))

    def grad_single_k(self, xi, yi, use_J=False):
        """Grad of kernel at a single point"""
        if use_J:
            J = self.J
        else:
            J = torch.eye(self.D_in)
        return J @ functorch.grad(self.single_k, argnums=0)(xi, yi)

    def grad_K(self, X, X2=None, use_J=False):
        """Grad of kernel at a set of points"""
        N1D = X.shape[0] * X.shape[1]
        N2 = X.shape
        if X2 is not None:
            N2 = X2.shape[0]
        if X2 is None:
            X2 = X
        return (
            functorch.vmap(
                lambda ti: functorch.vmap(
                    lambda tpi: self.grad_single_k(tpi, ti, use_J=use_J)
                )(X)
            )(X2)
            .permute(2, 1, 0)
            .reshape(N1D, N2)
        )

    def hess_single_k(self, x, xp, use_J=False):
        """Hessian of kernel at a single point"""
        if use_J:
            J = self.J
        else:
            J = torch.eye(self.D_in)
        return -J @ functorch.hessian(self.single_k)(x, xp) @ J.T

    def hess_K(self, X, X2=None, use_J=False):
        """Hessian of kernel at a set of points"""
        if X2 is not None:
            raise NotImplementedError

        ND = X.shape[0] * X.shape[1]
        return (
            functorch.vmap(
                lambda ti: functorch.vmap(
                    lambda tpi: self.hess_single_k(ti, tpi, use_J=use_J)
                )(X)
            )(X)
            .permute(2, 0, 3, 1)
            .reshape(ND, ND)
        )
