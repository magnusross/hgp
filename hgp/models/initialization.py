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

from hgp.misc import constraint_utils

import torch
import numpy as np
from scipy.cluster.vq import kmeans2

from hgp.core.kernels import DerivativeRBF, RBF
from scipy.signal import savgol_filter
from hgp.models.sequence import SequenceModel, UniformShootingModel
from hgp.core.dsvgp import DSVGP_Layer, Hamiltonian_DSVGP_Layer
from plum import dispatch


class MOExactHamiltonian(torch.nn.Module):
    def __init__(self, X, Fx, Z, init_ls=1.3, init_var=0.5, init_noise=1e-1):
        super().__init__()
        self.X = X
        self.D = X.shape[1]
        self.N = X.shape[0]
        self.M = Z.shape[0]
        self.Fx = Fx
        self.Z = Z

        self.kern = DerivativeRBF(X.shape[1], init_ls=init_ls, init_var=init_var)
        self.log_noise = torch.nn.Parameter(torch.log(torch.tensor(init_noise)))

    def construct(self):
        Ix = torch.eye(self.N * self.D) * torch.exp(self.log_noise)
        self.Kxx = self.kern.hess_K(self.X, use_J=True)  # (2DN,2DN)
        self.Kxz = self.kern.grad_K(self.X, self.Z, use_J=True)  #  (M, 2DN)
        Iz = torch.eye(self.M) * 1e-6
        self.Kzz = self.kern.K(self.Z)
        self.Lxx = torch.linalg.cholesky(self.Kxx + Ix)  # (N,N) or (D,N,N)
        self.Lzz = torch.linalg.cholesky(self.Kzz + Iz)

    def posterior_mean(self, whiten=True):
        self.construct()
        alpha = torch.linalg.solve_triangular(self.Lxx, self.Fx.T, upper=False)  # (N,D)
        alpha = torch.linalg.solve_triangular(self.Lxx.T, alpha, upper=True)  # (N,D)
        f_update = torch.einsum("nm, nd -> md", self.Kxz, alpha)  # (M,D)
        if whiten:
            return (
                torch.linalg.solve_triangular(
                    self.Lzz, f_update.T.unsqueeze(2), upper=False
                )
                .squeeze(2)
                .T
            )
        else:
            return f_update


@dispatch
def initialize_inducing(
    diffeq: Hamiltonian_DSVGP_Layer, data_ys, data_ts, data_noise=1e-1
):
    """
    Initialization of inducing variabels for Hamiltonian DSVGP layer.
    Inducing locations are initialized at cluster centers
    Inducing values are initialized using empirical data gradients.

    @param diffeq: a GP layer represnting the differential function
    @param data_ys: observed sequence (N,T,D)
    @param data_ts: data observation times, assumed to be equally spaced
    @param data_noise: an initial guess for observation noise.
    @return: the GP object after initialization
    """

    # compute empirical gradients and scale them according to observation time.
    f_xt = np.gradient(data_ys, data_ts, axis=1)
    f_xt = f_xt.reshape(-1, data_ys.shape[-1])  # (N,T-1,D)
    data_ys = data_ys[:, :-1, :]  # (N,T-1,D)
    data_ys = data_ys.reshape(-1, data_ys.shape[-1])  # (N*T-1,D)

    with torch.no_grad():
        num_obs_for_initialization = np.minimum(1000, data_ys.shape[0])
        obs_index = np.random.choice(
            data_ys.shape[0], num_obs_for_initialization, replace=False
        )

        inducing_loc = torch.tensor(
            kmeans2(data_ys, k=diffeq.Um().shape[0], minit="points")[0]
        )
        data_ys = torch.tensor(data_ys[obs_index])
        f_xt = torch.tensor(f_xt[obs_index].T.reshape(1, -1))

        pre_model = MOExactHamiltonian(
            data_ys,
            f_xt,
            inducing_loc,
            init_noise=0.1,
            init_ls=2.0,
            init_var=0.5,
        )

        pre_model.construct()
        inducing_val = pre_model.posterior_mean(whiten=True)

        diffeq.inducing_loc().data = inducing_loc.data  # (M,D)
        diffeq.Um().data = inducing_val.data  # (M,D)
        diffeq.kern.lengthscales = pre_model.kern.lengthscales.detach()
        diffeq.kern.variance = pre_model.kern.variance.detach()
        return diffeq


def compute_gpode_intial_inducing(kern, N_u, data_ys, empirical_fs, data_noise=1e-1):
    """
    Constructs initial inducing values using the process described in appendix of
    "Bayesian inference of ODEs with Gaussian processes", Hegde et al., 2021
    """
    # compute empirical gradients and scale them according to observation time.
    empirical_fs = empirical_fs.reshape(-1, empirical_fs.shape[-1])  # (N,T-1,D)
    data_ys = data_ys[:, :-1, :]  # (N,T-1,D)

    data_ys = data_ys.reshape(-1, data_ys.shape[-1])  # (N*T-1,D)

    with torch.no_grad():
        num_obs_for_initialization = np.minimum(1000, data_ys.shape[0])
        obs_index = np.random.choice(
            data_ys.shape[0], num_obs_for_initialization, replace=False
        )

        inducing_loc = torch.tensor(kmeans2(data_ys, k=N_u, minit="points")[0])
        data_ys = torch.tensor(data_ys[obs_index])
        empirical_fs = torch.tensor(empirical_fs[obs_index])

        Kxx = kern.K(data_ys)  # (N,N) or (D,N,N)
        Kxz = kern.K(data_ys, inducing_loc)  # (N,M) or (D,N,M)
        Kzz = kern.K(inducing_loc)  # (M,M) or (D,M,M)
        Lxx = torch.linalg.cholesky(
            Kxx + torch.eye(Kxx.shape[1]) * data_noise
        )  # (N,N) or (D,N,N)
        Lzz = torch.linalg.cholesky(
            Kzz + torch.eye(Kzz.shape[1]) * 1e-6
        )  # (M,M) or (D,M,M)

        if not kern.dimwise:
            alpha = torch.linalg.solve_triangular(
                Lxx, empirical_fs, upper=False
            )  # (N,D)
            alpha = torch.linalg.solve_triangular(Lxx.T, alpha, upper=True)  # (N,D)
            f_update = torch.einsum("nm, nd -> md", Kxz, alpha)  # (M,D)
        else:
            alpha = torch.linalg.solve_triangular(
                Lxx, empirical_fs.T.unsqueeze(2), upper=False
            )  # (N,D)
            alpha = torch.linalg.solve_triangular(
                Lxx.permute(0, 2, 1), alpha, upper=True
            )  # (N,D)
            f_update = torch.einsum("dnm, dn -> md", Kxz, alpha.squeeze(2))  # (M,D)

        inducing_val = (
            torch.linalg.solve_triangular(Lzz, f_update.T.unsqueeze(2), upper=False)
            .squeeze(2)
            .T
        )  # (M,D)
        return inducing_loc.data, inducing_val.data


@dispatch
def initialize_inducing(diffeq: DSVGP_Layer, data_ys, data_ts, data_noise=1e-1):
    """
    Initialization of inducing variabels for standard DSVGP layer.
    Inducing locations are initialized at cluster centers
    Inducing values are initialized using empirical data gradients.

    @param diffeq: a GP layer represnting the differential function
    @param data_ys: observed sequence (N,T,D)
    @param data_noise: an initial guess for observation noise.
    @return: the gp object after initialization
    """
    empirical_fs = np.gradient(data_ys, data_ts, axis=1)  # (N,T-1,D)
    inducing_loc, inducing_val = compute_gpode_intial_inducing(
        diffeq.kern,
        diffeq.Um().shape[0],
        data_ys,
        empirical_fs,
    )
    diffeq.inducing_loc().data = inducing_loc  # (M,D)
    diffeq.Um().data = inducing_val  # (M,D)
    return diffeq


def initialize_latents_with_data(model, data_ys, data_ts, num_samples=20):
    """
    Initializes shooting states from data.
    Initial state distribution is initialized by solving the ODE backward in time from the first
        observation after inducing variables are initialized.
    Other states are initialized at observed values.

    @param model: a gpode.UniformShootingModel object
    @param data_ys: observed state sequence
    @param num_samples: number of samples to consider for initial state initialization
    @return: the model object after initialization
    """
    with torch.no_grad():
        # this makes sure we only take the data points that align with our shooting states
        try:
            init_xs = torch.tensor(
                data_ys[:, 0 : -model.shooting_time_factor : model.shooting_time_factor]
            )
        except AttributeError:
            init_xs = torch.tensor(data_ys[:, :-1])

        ts = torch.tensor(data_ts)
        init_ts = torch.cat([ts[1:2], ts[0:1]])
        init_x0 = []
        for _ in range(num_samples):
            init_x0.append(
                model.build_flow(init_xs[:, 0], init_ts).clone().detach().data[:, -1]
            )
        init_x0 = torch.stack(init_x0).mean(0)
        model.state_distribution._initialize(init_x0, init_xs)
    return model


def initalize_noisevar(model, init_noisevar):
    """
    Initializes likelihood observation noise variance parameter

    @param model: a gpode.SequenceModel object
    @param init_noisevar: initialization value
    @return: the model object after initialization
    """
    model.observation_likelihood.unconstrained_variance.data = (
        constraint_utils.invsoftplus(torch.tensor(init_noisevar)).data
    )
    return model


def initialize_and_fix_kernel_parameters(
    model, lengthscale_value=1.25, variance_value=0.5, fix=False
):
    """
    Initializes and optionally fixes kernel parameter

    @param model: a gpode.SequenceModel object
    @param lengthscale_value: initialization value for kernel lengthscales parameter
    @param variance_value: initialization value for kernel signal variance parameter
    @param fix: a flag variable to fix kernel parameters during optimization
    @return: the model object after initialization
    """
    model.flow.odefunc.diffeq.kern.unconstrained_lengthscales.data = (
        constraint_utils.invsoftplus(
            lengthscale_value
            * torch.ones_like(
                model.flow.odefunc.diffeq.kern.unconstrained_lengthscales.data
            )
        )
    )
    model.flow.odefunc.diffeq.kern.unconstrained_variance.data = (
        constraint_utils.invsoftplus(
            variance_value
            * torch.ones_like(
                model.flow.odefunc.diffeq.kern.unconstrained_variance.data
            )
        )
    )
    if fix:
        model.flow.odefunc.diffeq.kern.unconstrained_lengthscales.requires_grad_(False)
        model.flow.odefunc.diffeq.kern.unconstrained_variance.requires_grad_(False)
    return model
