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

import copy
import time
from typing import Union

import torch
from plum import dispatch
from torch.utils.data import DataLoader, Dataset

import hgp

# from scipy.stats import norm
# from scipy.special import logsumexp
import hgp.misc.metrics as metrics
from hgp.core import constraint_likelihoods as constraints
from hgp.core.dsvgp import DSVGP_Layer, Hamiltonian_DSVGP_Layer
from hgp.core.flow import Flow
from hgp.core.nn import HamiltonianNNModel, NNModel
from hgp.core.observation_likelihoods import Gaussian
from hgp.core.states import (
    DeltaStateSequenceDistribution,
    StateInitialVariationalGaussian,
    StateSequenceVariationalFactorizedGaussian,
)
from hgp.misc import train_utils as utils
from hgp.misc.torch_utils import insert_zero_t0, numpy2torch, torch2numpy
from hgp.models.initialization import initialize_inducing, initialize_latents_with_data
from hgp.models.sequence import (
    ConsUniformShootingModel,
    NNSequenceModel,
    NNUniformShootingModel,
    SequenceModel,
    SubSequenceModel,
    UniformShootingModel,
)


@dispatch
def init_and_fit(
    model: Union[UniformShootingModel, SequenceModel],
    args,
    data_ts,
    data_ys,
    return_history=False,
):

    if args.model.inducing_init:
        model.flow.odefunc.diffeq = initialize_inducing(
            model.flow.odefunc.diffeq, data_ys, data_ts
        )

    model = initialize_latents_with_data(model, data_ys, data_ts)

    trainer = Trainer()
    model, history = trainer.train(
        model=model,
        loss_function=compute_loss,
        ys=numpy2torch(data_ys),
        ts=numpy2torch(data_ts),
        num_iter=args.num_iter,
        lr=args.lr,
        log_freq=args.log_freq,
    )
    if return_history:
        return model, history
    else:
        return model


@dispatch
def init_and_fit(
    model: SubSequenceModel,
    args,
    data_ts,
    data_ys,
    return_history=False,
):

    if args.model.inducing_init:
        model.flow.odefunc.diffeq = initialize_inducing(
            model.flow.odefunc.diffeq, data_ys, data_ts
        )

    trainer = BatchedTrainer()
    model, history = trainer.train(
        model=model,
        loss_function=compute_loss,
        ys=numpy2torch(data_ys),
        ts=numpy2torch(data_ts),
        num_iter=args.num_iter,
        lr=args.lr,
        log_freq=args.log_freq,
    )
    if return_history:
        return model, history
    else:
        return model


@dispatch
def init_and_fit(
    model: NNUniformShootingModel,
    args,
    data_ts,
    data_ys,
    return_history=False,
):

    model = initialize_latents_with_data(model, data_ys, data_ts)
    trainer = NNTrainer()
    model, history = trainer.train(
        model=model,
        loss_function=compute_loss,
        ys=numpy2torch(data_ys),
        ts=numpy2torch(data_ts),
        num_iter=args.num_iter,
        lr=args.lr,
        log_freq=args.log_freq,
    )
    if return_history:
        return model, history
    else:
        return model


@dispatch
def init_and_fit(
    model: NNSequenceModel,
    args,
    data_ts,
    data_ys,
    return_history=False,
):

    trainer = BatchedNNTrainer()
    model, history = trainer.train(
        model=model,
        loss_function=compute_loss,
        ys=numpy2torch(data_ys),
        ts=numpy2torch(data_ts),
        num_iter=args.num_iter,
        lr=args.lr,
        log_freq=args.log_freq,
        batch_length=args.model.batch_length,
        batch_size=args.model.batch_size,
        num_val_epochs=args.model.num_val_epochs,
    )
    if return_history:
        return model, history
    else:
        return model


def build_model(args, data_ys):
    """
    Builds a HGP model based on training sequence

    @param args: model setup arguments
    @param data_ys: observed/training sequence of (N,T,D) dimensions
    """
    N, T, D = data_ys.shape

    gp = Hamiltonian_DSVGP_Layer(
        D_in=D,
        M=args.model.num_inducing,
        S=args.model.num_features,
        q_diag=args.model.q_diag,
    )

    flow = Flow(diffeq=gp, solver=args.solver, use_adjoint=args.use_adjoint)

    observation_likelihood = Gaussian(ndim=D, init_val=args.init_noise)

    if args.model.shooting:
        if args.model.shooting_time_factor is None:
            args.model.shooting_time_factor = 1

        assert (
            (T - 1) % args.model.shooting_time_factor
        ) == 0, f"T-1 must be devisable by time factor, T={T}, F={args.model.shooting_time_factor}"

        N_shooting = (T - 1) // args.model.shooting_time_factor

        if args.model.constraint_type not in ["gauss", "laplace"]:
            raise ValueError(
                "invalid constraint likelihood specification, only available options are gauss/laplace"
            )

        constraint_type_class = (
            constraints.Laplace
            if args.model.constraint_type == "laplace"
            else constraints.Gaussian
        )
        constraint_likelihood = constraint_type_class(
            d=1,
            scale=args.model.constraint_initial_scale,
            requires_grad=args.model.constraint_trainable,
        )
        energy_likelihood = (
            constraint_type_class(
                d=1,
                scale=args.model.energy_constraint_initial_scale,
                requires_grad=args.model.constraint_trainable,
            )
            if args.model.constrain_energy
            else None
        )

        model = (
            ConsUniformShootingModel
            if args.model.constrain_energy
            else UniformShootingModel
        )(
            flow=flow,
            num_observations=N * T * D,
            state_distribution=StateSequenceVariationalFactorizedGaussian(
                dim_n=N, dim_t=N_shooting, dim_d=D
            ),
            observation_likelihood=observation_likelihood,
            constraint_likelihood=constraint_likelihood,
            shooting_time_factor=args.model.shooting_time_factor,
            energy_likelihood=energy_likelihood,
            ts_dense_scale=args.model.ts_dense_scale,
        )
    else:
        model = SequenceModel(
            flow=flow,
            num_observations=N * T * D,
            state_distribution=StateInitialVariationalGaussian(dim_n=N, dim_d=D),
            observation_likelihood=observation_likelihood,
            constraint_likelihood=None,
            ts_dense_scale=args.model.ts_dense_scale,
        )

    return model


def build_subsequence_model(args, data_ys):
    """
    Builds a HGP-Batched model based on training sequence

    @param args: model setup arguments
    @param data_ys: observed/training sequence of (N,T,D) dimensions
    """
    N, T, D = data_ys.shape

    gp = Hamiltonian_DSVGP_Layer(
        D_in=D,
        M=args.model.num_inducing,
        S=args.model.num_features,
        q_diag=args.model.q_diag,
    )

    flow = Flow(diffeq=gp, solver=args.solver, use_adjoint=args.use_adjoint)

    observation_likelihood = Gaussian(ndim=D, init_val=args.init_noise)

    model = SubSequenceModel(
        flow=flow,
        num_observations=N * T * D,
        state_distribution=None,
        observation_likelihood=observation_likelihood,
        constraint_likelihood=None,
        ts_dense_scale=args.model.ts_dense_scale,
    )

    return model


def build_gpode_model(args, data_ys):
    """
    Builds a GP-ODE model based on training sequence

    @param args: model setup arguments
    @param data_ys: observed/training sequence of (N,T,D) dimensions
    """
    N, T, D = data_ys.shape

    gp = DSVGP_Layer(
        D_in=D,
        D_out=D,
        M=args.model.num_inducing,
        S=args.model.num_features,
        dimwise=args.model.dimwise,
        q_diag=args.model.q_diag,
    )

    flow = Flow(diffeq=gp, solver=args.solver, use_adjoint=args.use_adjoint)

    observation_likelihood = Gaussian(ndim=D, init_val=args.init_noise)

    if args.model.shooting:
        if args.model.shooting_time_factor is None:
            args.model.shooting_time_factor = 1

        assert (
            (T - 1) % args.model.shooting_time_factor
        ) == 0, "T-1 must be devisable by time factor"

        N_shooting = (T - 1) // args.model.shooting_time_factor

        if args.model.constraint_type not in ["gauss", "laplace"]:
            raise ValueError(
                "invalid constraint likelihood specification, only available options are gauss/laplace"
            )

        constraint_type_class = (
            constraints.Laplace
            if args.model.constraint_type == "laplace"
            else constraints.Gaussian
        )
        constraint_likelihood = constraint_type_class(
            d=1,
            scale=args.model.constraint_initial_scale,
            requires_grad=args.model.constraint_trainable,
        )

        model = UniformShootingModel(
            flow=flow,
            num_observations=N * T * D,
            state_distribution=StateSequenceVariationalFactorizedGaussian(
                dim_n=N, dim_t=N_shooting, dim_d=D
            ),
            observation_likelihood=observation_likelihood,
            constraint_likelihood=constraint_likelihood,
            shooting_time_factor=args.model.shooting_time_factor,
            ts_dense_scale=args.model.ts_dense_scale,
        )
    else:
        model = SequenceModel(
            flow=flow,
            num_observations=N * T * D,
            state_distribution=StateInitialVariationalGaussian(dim_n=N, dim_d=D),
            observation_likelihood=observation_likelihood,
            constraint_likelihood=None,
            ts_dense_scale=args.model.ts_dense_scale,
        )

    return model


def build_nn_model(args, data_ys):
    """
    Builds a NN model based on training sequence

    @param args: model setup arguments
    @param data_ys: observed/training sequence of (N,T,D) dimensions
    """
    N, T, D = data_ys.shape

    if args.model.flow_type == "hnn":
        nn = HamiltonianNNModel(
            D_in=D,
            N_layers=args.model.N_layers,
            N_nodes=args.model.N_nodes,
        )

    elif args.model.flow_type == "node":
        nn = NNModel(
            D_in=D,
            D_out=D,
            N_layers=args.model.N_layers,
            N_nodes=args.model.N_nodes,
        )
    else:
        raise ValueError

    flow = Flow(diffeq=nn, solver=args.solver, use_adjoint=args.use_adjoint)
    if args.model.shooting:
        model = NNUniformShootingModel(
            flow=flow,
            num_observations=N * T * D,
            state_distribution=DeltaStateSequenceDistribution(
                dim_n=N, dim_t=T - 1, dim_d=D
            ),
            observation_likelihood=torch.nn.L1Loss(),
            constraint_likelihood=torch.nn.L1Loss(),
            shooting_time_factor=args.model.shooting_time_factor,
            ts_dense_scale=args.model.ts_dense_scale,
            alpha=args.model.alpha,
        )
    else:
        model = NNSequenceModel(
            flow=flow,
            num_observations=N * T * D,
            state_distribution=None,
            observation_likelihood=torch.nn.L1Loss(),
            constraint_likelihood=None,
            ts_dense_scale=args.model.ts_dense_scale,
        )
    # hack to get likelihoods to show nan as the model doesn't have one
    model.observation_likelihood.variance = torch.tensor(torch.nan)
    return model


@dispatch
def compute_loss(model: NNSequenceModel, ys, ts):
    """
    Compute loss for model optimization.


    @param model: a model object
    @param ys: true observation sequence
    @param ts: observation times
    @return: loss, nan, nan
    """
    loss = model.build_lowerbound_terms(ys, ts)
    return loss, torch.tensor(torch.nan), torch.tensor(torch.nan)


@dispatch
def compute_loss(model: NNUniformShootingModel, ys, ts):
    """
    Compute loss for model optimization.
    @param model: a model object
    @param ys: true observation sequence
    @param ts: observation times
    @return loss, observation loss, shooting loss
    """
    obs_loss, shooting_loss = model.build_lowerbound_terms(ys, ts)
    return obs_loss + shooting_loss, obs_loss, shooting_loss


@dispatch
def compute_loss(model: UniformShootingModel, ys, ts, **kwargs):
    """
    Compute loss for model optimization.
    @param model: a model object
    @param ys: true observation sequence
    @param ts: observation times
    @param kwargs: additional parameters passed to the model.build_lowerbound_terms() method
    @return: loss, nll, nan, initial_state_kl, inducing_kl
    """
    (
        observation_loglik,
        state_constraint_logpob,
        state_entropy,
        init_state_kl,
    ) = model.build_lowerbound_terms(ys, ts, **kwargs)
    inducing_kl = model.build_inducing_kl()
    loss = -(
        observation_loglik
        + state_constraint_logpob
        + state_entropy
        - init_state_kl
        - inducing_kl
    )
    return (
        loss,
        -observation_loglik,
        -(state_constraint_logpob),
        torch.tensor(torch.nan),
        init_state_kl,
        inducing_kl,
    )


@dispatch
def compute_loss(model: ConsUniformShootingModel, ys, ts, **kwargs):
    """
    Compute loss for model optimization.
    @param model: a model object
    @param ys: true observation sequence
    @param ts: observation times
    @param kwargs: additional parameters passed to the model.build_lowerbound_terms() method
    @return: loss, nll, energy constraint, initial_state_kl, inducing_kl
    """
    (
        observation_loglik,
        state_constraint_logpob,
        energy_constraint_logpob,
        state_entropy,
        init_state_kl,
    ) = model.build_lowerbound_terms(ys, ts, **kwargs)
    inducing_kl = model.build_inducing_kl()
    loss = -(
        observation_loglik
        + state_constraint_logpob
        + energy_constraint_logpob
        + state_entropy
        - init_state_kl
        - inducing_kl
    )
    return (
        loss,
        -observation_loglik,
        -(state_constraint_logpob),
        -(energy_constraint_logpob),
        init_state_kl,
        inducing_kl,
    )


@dispatch
def compute_loss(model: SequenceModel, ys, ts, **kwargs):
    """
    Compute loss for model optimization, no shooting.
    @param model: a model object
    @param ys: true observation sequence
    @param ts: observation times
    @return: loss, nll, nan, nan, initial_state_kl, inducing_kl
    """
    observ_loglik, init_state_kl = model.build_lowerbound_terms(ys, ts)
    kl = model.build_inducing_kl()
    loss = -(observ_loglik - init_state_kl - kl)
    return (
        loss,
        -observ_loglik,
        torch.tensor(torch.nan),
        torch.tensor(torch.nan),
        init_state_kl,
        kl,
    )


@dispatch
def compute_loss(model: SubSequenceModel, ys, ts, **kwargs):
    """
    Compute loss for model optimization, batched training.
    @param model: a gpode.SequenceModel object
    @param ys: true observation sequence
    @param ts: observation times
    @param kwargs: additional parameters passed to the model.build_lowerbound_terms() method
    @return: loss, nll, inducing_kl
    """
    observ_loglik = model.build_lowerbound_terms(ys, ts)
    kl = model.build_inducing_kl()
    loss = -(observ_loglik - kl)
    return (
        loss,
        -observ_loglik,
        kl,
    )


@dispatch
def compute_single_prediction(
    model: Union[
        UniformShootingModel, ConsUniformShootingModel, NNUniformShootingModel
    ],
    ts,
):
    """
    Computes single prediction from a model from an optimized initial state
    Useful while making predictions/extrapolation to novel time points from an optimized initial state.

    @param model: a model object
    @param ts: observation times
    @return: predictive samples
    """
    # add additional time point accounting the initial state
    ts = insert_zero_t0(ts)
    return model(model.state_distribution.x0.sample().squeeze(0), ts)


@dispatch
def compute_single_prediction(model: Union[SequenceModel, NNSequenceModel], ts):
    """
    Computes single prediction from a model from an optimized initial state
    Useful while making predictions/extrapolation to novel time points from an optimized initial state.

    @param model: a model object
    @param ts: observation times
    @return: predictive samples
    """
    # add additional time point accounting the initial state
    ts = insert_zero_t0(ts)
    return model(model.state_distribution.sample().squeeze(0), ts)


def compute_predictions(model, ts, eval_sample_size=10):
    """
    Compute predictions or ODE sequences from a GPODE model from an optimized initial state
    Useful while making predictions/extrapolation to novel time points from an optimized initial state.

    @param model: a model object
    @param ts: observation times
    @param eval_sample_size: number of samples for evaluation
    @return: predictive samples
    """
    model.eval()

    pred_samples = []
    for _ in range(eval_sample_size):
        with torch.no_grad():
            pred_samples.append(compute_single_prediction(model, ts))
    return torch.stack(pred_samples, 0)[:, :, 1:]


def compute_test_predictions(model, y0, ts, eval_sample_size=10):
    """
    Compute predictions or ODE sequences from a GPODE model from an given initial state

    @param model: a gpode.SequenceModel object
    @param y0: initial state for computing predictions (N,D)
    @param ts: observation times
    @param eval_sample_size: number of samples for evaluation
    @return: predictive samples
    """
    model.eval()
    pred_samples = []
    for _ in range(eval_sample_size):
        with torch.no_grad():
            pred_samples.append(model(y0, ts))

    return torch.stack(pred_samples, 0)


def compute_summary(actual, predicted, noise_var, ys=1.0, squeeze_time=True):
    """
    Computes MSE and MLL as summary metrics between actual and predicted sequences
    @param actual: true observation sequnce
    @param predicted: predicted sequence
    @param noise_var: noise var predicted by the model
    @param ys: optional scaling factor for standardized data
    @param squeeze_time: optional, if true averages over time dimension
    @return: MLL(actual, predicted),  MSE(actual, predicted)
    """
    actual = actual * ys
    predicted = predicted * ys
    noise_var = noise_var * ys**2 + 1e-8
    if squeeze_time:

        return (
            metrics.log_lik(actual, predicted, noise_var).mean(),
            metrics.mse(actual, predicted).mean(),
            metrics.rel_err(actual, predicted).mean(),
        )

    else:
        return (
            metrics.log_lik(actual, predicted, noise_var).mean(2),
            metrics.mse(actual, predicted).mean(2),
            metrics.rel_err(actual, predicted),
        )


class Trainer:
    """
    A trainer class for models. Stores optimization trace for monitoring/plotting purpose
    """

    def __init__(self):
        self.loss_meter = utils.CachedRunningAverageMeter(0.98)
        self.observation_nll_meter = utils.CachedRunningAverageMeter(0.98)
        self.state_kl_meter = utils.CachedRunningAverageMeter(0.98)
        self.energy_kl_meter = utils.CachedRunningAverageMeter(0.98)
        self.init_kl_meter = utils.CachedRunningAverageMeter(0.98)
        self.inducing_kl_meter = utils.CachedRunningAverageMeter(0.98)
        self.time_meter = utils.CachedAverageMeter()
        self.compute_loss = compute_loss

    def train(self, model, loss_function, ys, ts, num_iter, lr, log_freq, **kwargs):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        print("Fitting model...")
        for itr in range(1, num_iter):
            try:
                model.train()
                begin = time.time()
                optimizer.zero_grad()

                (
                    loss,
                    observation_nll,
                    state_kl,
                    energy_kl,
                    init_kl,
                    inducing_kl,
                ) = loss_function(model, ys, ts, **kwargs)

                loss.backward()
                optimizer.step()

                self.loss_meter.update(loss.item(), itr)
                self.observation_nll_meter.update(observation_nll.item(), itr)
                self.state_kl_meter.update(state_kl.item(), itr)
                self.energy_kl_meter.update(energy_kl.item(), itr)
                self.init_kl_meter.update(init_kl.item(), itr)
                self.inducing_kl_meter.update(inducing_kl.item(), itr)
                self.time_meter.update(time.time() - begin, itr)

                if itr % log_freq == 0:
                    log_message = (
                        "Iter {:04d} | Loss {:.2f}({:.2f}) |"
                        "OBS NLL {:.2f}({:.2f}) | XS KL {:.2f}({:.2f}) |"
                        " E KL {:.2f}({:.2f}) |"
                        "X0 KL {:.2f}({:.2f}) | IND KL {:.2f}({:.2f})".format(
                            itr,
                            self.loss_meter.val,
                            self.loss_meter.avg,
                            self.observation_nll_meter.val,
                            self.observation_nll_meter.avg,
                            self.state_kl_meter.val,
                            self.state_kl_meter.avg,
                            self.energy_kl_meter.val,
                            self.energy_kl_meter.avg,
                            self.init_kl_meter.val,
                            self.init_kl_meter.avg,
                            self.inducing_kl_meter.val,
                            self.inducing_kl_meter.avg,
                        )
                    )
                    print(log_message)

            except KeyboardInterrupt:
                break
        return model, self


class MultiDataset(Dataset):
    def __init__(self, ts, ys):
        self.ts = ts
        self.ys = ys

        self.len = ys.shape[0]
        self.batch_length = ys.shape[1]
        self.d = ys.shape[-1]

    def __getitem__(self, index):
        return self.ys[index]

    def __len__(self):
        return self.len


class BatchedTrainer:
    """
    A trainer class for batched models. Stores optimization trace for monitoring/plotting purpose
    """

    def __init__(self):
        self.loss_meter = utils.CachedRunningAverageMeter(0.98)
        self.observation_nll_meter = utils.CachedRunningAverageMeter(0.98)
        self.state_kl_meter = utils.CachedRunningAverageMeter(0.98)

    def train(
        self,
        model,
        loss_function,
        ys,
        ts,
        num_iter,
        lr,
        log_freq,
        batch_length=10,
        batch_size=32,
        num_val_epochs=10,
        **kwargs,
    ):

        if type(model) != SubSequenceModel:
            raise ValueError("Batch training only supported for SubSequenceModel")
        # only single examples for now
        batch_ys = torch.stack(
            [
                ys[:, i : i + batch_length]
                for i in range(ys.shape[1] - batch_length + 1)
            ],
            axis=0,
        )
        batch_ys = batch_ys.reshape(-1, batch_ys.shape[2], batch_ys.shape[-1])
        batch_ts = ts[:batch_length]

        dataset = MultiDataset(batch_ts, batch_ys)
        trainloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        i = 0
        best_metric = 1e6
        best_model = copy.deepcopy(model)

        print("Fitting model...")
        for itr in range(1, 100000):
            try:
                for ysi in trainloader:
                    i += 1

                    model.train()
                    optimizer.zero_grad()
                    # print(ysi.shape)

                    # ysi = ysi.reshape(-1, dataset.batch_length, dataset.d)

                    loss, obs_like, initial_kl = loss_function(
                        model, ysi, dataset.ts, **kwargs
                    )

                    loss.backward()
                    optimizer.step()

                    self.loss_meter.update(loss.item(), i)
                    self.state_kl_meter.update(initial_kl.item(), i)
                    self.observation_nll_meter.update(obs_like.item(), i)

            except KeyboardInterrupt:
                break

            if num_val_epochs:
                if itr % num_val_epochs == 0:
                    preds = hgp.models.builder.compute_test_predictions(
                        model,
                        numpy2torch(ys[:, 0, :]),
                        numpy2torch(ts),
                        eval_sample_size=2,
                    )
                    mll, _, _ = hgp.models.builder.compute_summary(
                        torch2numpy(ys),
                        torch2numpy(preds),
                        torch2numpy(model.observation_likelihood.variance),
                    )
                    mnll = -mll

                    if mnll < best_metric:
                        best_model = copy.deepcopy(model)
                        best_metric = mnll

            if itr % log_freq == 0:
                log_message = "Iter {:04d} | Loss {:.3f}({:.3f}) | OBS {:.3f}({:.3f}) | KL {:.3f}({:.3f}) | Best Metric {:.3f}".format(
                    i,
                    self.loss_meter.val,
                    self.loss_meter.avg,
                    self.observation_nll_meter.val,
                    self.observation_nll_meter.avg,
                    self.state_kl_meter.val,
                    self.state_kl_meter.avg,
                    best_metric,
                )
                print(log_message)
            if i > num_iter:
                break

        return best_model, self


class BatchedNNTrainer:
    """
    A trainer class for batched NN models. Stores optimization trace for monitoring/plotting purpose
    """

    def __init__(self):
        self.loss_meter = utils.CachedRunningAverageMeter(0.98)
        self.observation_nll_meter = utils.CachedRunningAverageMeter(0.98)
        self.state_kl_meter = utils.CachedRunningAverageMeter(0.98)

    def train(
        self,
        model,
        loss_function,
        ys,
        ts,
        num_iter,
        lr,
        log_freq,
        batch_length=5,
        batch_size=32,
        num_val_epochs=10,
        **kwargs,
    ):

        # only single examples for now
        batch_ys = torch.stack(
            [
                ys[:, i : i + batch_length]
                for i in range(ys.shape[1] - batch_length + 1)
            ],
            axis=0,
        )
        batch_ys = batch_ys.reshape(-1, batch_ys.shape[2], batch_ys.shape[-1])
        batch_ts = ts[:batch_length]

        dataset = MultiDataset(batch_ts, batch_ys)
        trainloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        i = 0
        best_metric = 1e6
        best_model = copy.deepcopy(model)

        print("Fitting model...")
        for itr in range(1, 100000):
            try:
                for ysi in trainloader:
                    i += 1

                    model.train()
                    optimizer.zero_grad()
                    # print(ysi.shape)

                    loss, obs_loss, shooting_loss = loss_function(
                        model, ysi, dataset.ts, **kwargs
                    )

                    loss.backward()
                    optimizer.step()

                    self.loss_meter.update(loss.item(), i)
                    self.state_kl_meter.update(obs_loss.item(), i)
                    self.observation_nll_meter.update(shooting_loss.item(), i)

            except KeyboardInterrupt:
                break

            if num_val_epochs:
                if itr % num_val_epochs == 0:
                    preds = hgp.models.builder.compute_test_predictions(
                        model,
                        numpy2torch(ys[:, 0, :]),
                        numpy2torch(ts),
                        eval_sample_size=1,
                    )
                    _, mse, _ = hgp.models.builder.compute_summary(
                        torch2numpy(ys),
                        torch2numpy(preds),
                        torch2numpy(model.observation_likelihood.variance),
                    )

                    if mse < best_metric:
                        best_model = copy.deepcopy(model)
                        best_metric = mse

            if itr % log_freq == 0:
                log_message = "Iter {:04d} | Loss {:.3f}({:.3f}) | OBS {:.3f}({:.3f}) | KL {:.3f}({:.3f}) | Best Metric {:.3f}".format(
                    i,
                    self.loss_meter.val,
                    self.loss_meter.avg,
                    self.observation_nll_meter.val,
                    self.observation_nll_meter.avg,
                    self.state_kl_meter.val,
                    self.state_kl_meter.avg,
                    best_metric,
                )
                print(log_message)
            if i > num_iter:
                break

        return best_model, self


class NNTrainer:
    """
    A trainer class for NN models. Stores optimization trace for monitoring/plotting purpose
    """

    def __init__(self):
        self.loss_meter = utils.CachedRunningAverageMeter(0.98)
        self.observation_nll_meter = utils.CachedRunningAverageMeter(0.98)
        self.state_kl_meter = utils.CachedRunningAverageMeter(0.98)

    def train(self, model, loss_function, ys, ts, num_iter, lr, log_freq, **kwargs):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        print("Fitting model...")
        for itr in range(1, num_iter):
            try:
                model.train()
                begin = time.time()
                optimizer.zero_grad()

                loss, obs_loss, shooting_loss = loss_function(model, ys, ts, **kwargs)

                loss.backward()
                optimizer.step()

                self.loss_meter.update(loss.item(), itr)
                self.observation_nll_meter.update(obs_loss.item(), itr)
                self.state_kl_meter.update(shooting_loss.item(), itr)

                if itr % log_freq == 0:
                    log_message = "Iter {:04d} | Loss {:.3f}({:.3f}) | Obs {:.3f}({:.3f}) | Shooting {:.3f}({:.3f}) |".format(
                        itr,
                        self.loss_meter.val,
                        self.loss_meter.avg,
                        self.observation_nll_meter.val,
                        self.observation_nll_meter.avg,
                        self.state_kl_meter.val,
                        self.state_kl_meter.avg,
                    )
                    print(log_message)

            except KeyboardInterrupt:
                break
        return model, self
