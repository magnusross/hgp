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

from hgp.misc.torch_utils import insert_zero_t0, compute_ts_dense

from torch import nn
import torch


def stack_segments(unstacked):
    return unstacked.reshape(-1, unstacked.shape[-1])


def unstack_segments(stacked, unstacked_shape):
    return stacked.reshape(unstacked_shape)


class BaseSequenceModel(nn.Module):
    """
    Implements base class for model for learning unknown Hamiltonian system.
    Model setup for observations on non-uniform grid or mini-batching over time can be derived from this class.

    Defines following methods:
        build_flow: given an initial state and time sequence, perform forward ODE integration
        build_flow_and_divergence: performs coupled forward ODE integration for states and density change
        build_lowerbound_terms: given observed states and time, builds individual terms for the lowerbound computation
        build_inducing_kl: computes KL divergence between inducing prior and posterior.
        forward: a wrapper for build_flow method
    """

    def __init__(
        self,
        flow,
        num_observations,
        state_distribution,
        observation_likelihood,
        constraint_likelihood,
        ts_dense_scale=2,
    ):
        super(BaseSequenceModel, self).__init__()
        self.flow = flow
        self.num_observations = num_observations
        self.state_distribution = state_distribution

        self.observation_likelihood = observation_likelihood
        self.constraint_likelihood = constraint_likelihood
        self.ts_dense_scale = ts_dense_scale

    def build_flow(self, x0, ts):
        """
        Given an initial state and time sequence, perform forward ODE integration
        Optionally, the time sequnce can be made dense based on self.ts_dense_scale prameter

        @param x0: initial state tensor (N,D)
        @param ts: time sequence tensor (T,)
        @return: forward solution tensor (N,T,D)
        """
        ts = compute_ts_dense(ts, self.ts_dense_scale)
        ys = self.flow(x0, ts, return_energy=False)
        return ys[:, :: self.ts_dense_scale - 1, :]

    def build_lowerbound_terms(self, ys, ts, **kwargs):
        raise NotImplementedError

    def build_objective(self, ys, ts):
        raise NotImplementedError

    def build_inducing_kl(self):
        """
        Computes KL divergence between inducing prior and posterior.

        @return: inducing KL scaled by the number of observations
        """
        return self.flow.kl() / self.num_observations

    def forward(self, x0, ts):
        """
        A wrapper for build_flow method
        @param x0: initial state tensor (N,D)
        @param ts: time sequence tensor (T,)
        @return: forward solution tensor (N,T,D)
        """
        return self.build_flow(x0, ts)


class NNSequenceModel(BaseSequenceModel):
    """
    Implements sequence model for neural network derivative functions.
    """

    def build_objective(self, ys, ts):
        raise NotImplementedError

    def build_inducing_kl(self):
        raise NotImplementedError

    def build_lowerbound_terms(self, ys_batched, ts, num_samples=1):
        """
        Given oberved states and time, builds the individual terms for the lowerbound computation

        @param ys: observed sequence tensor (N,T,D)
        @param ts: observed time sequence (T,)
        @return: nll, initial state KL
        """
        assert num_samples == 1, ">1 sample not implemented for standard model."

        xs = self.build_flow(ys_batched[:, 0, :], ts)
        mse = self.observation_likelihood(xs, ys_batched)
        # print(mse.shape)
        return mse


class NNUniformShootingModel(BaseSequenceModel):
    """
    Neural network model, with shooting.
    """

    def __init__(
        self,
        flow,
        num_observations,
        state_distribution,
        observation_likelihood,
        constraint_likelihood,
        shooting_time_factor=None,
        ts_dense_scale=2,
        alpha=100,
    ):
        super(NNUniformShootingModel, self).__init__(
            flow=flow,
            num_observations=num_observations,
            state_distribution=state_distribution,
            observation_likelihood=observation_likelihood,
            constraint_likelihood=constraint_likelihood,
            ts_dense_scale=ts_dense_scale,
        )

        self.shooting_time_factor = shooting_time_factor
        self.alpha = alpha

    def build_objective(self, ys, ts):
        loss, shooting_loss = self.build_lowerbound_terms(ys, ts)
        return loss + shooting_loss

    def build_inducing_kl(self):
        raise NotImplementedError

    def build_lowerbound_terms(self, ys, ts, num_samples=1):
        """
        Given oberved states and time, builds the individual terms for the lowerbound computation

        @param ys: observed sequence tensor (N,T,D)
        @param ts: observed time sequence (T,)
        @return: nll, initial state KL
        """
        assert num_samples == 1, ">1 sample not implemented for nn model."
        ss_samples = self.state_distribution.sample(
            num_samples=num_samples
        )  # (S,N,(T-1)/shooting_time_factor + 1, D)
        (S, N, N_state, D) = ss_samples.shape
        N_shooting = N_state - 1
        T = ts.shape[0]
        shooting_time_factor = (T - 1) // N_shooting

        assert (
            shooting_time_factor == self.shooting_time_factor
        ), f"{shooting_time_factor}, {T}, {N_shooting}"

        predicted_xs = self.flow(
            x0=stack_segments(ss_samples[:, :, 1:, :]),
            ts=ts[: shooting_time_factor + 1],
        )  # (SxNxN_state, shooting_time_factor+1, D)
        predicted_xs = unstack_segments(
            predicted_xs[:, 1:], (S, N, T - 1, D)
        )  # (S, N, T-1, D)

        predicted_x0 = self.flow(
            x0=stack_segments(ss_samples[:, :, 0, :]),
            ts=ts[:2],
        )
        predicted_x0 = unstack_segments(predicted_x0[:, -1], (S, N, 1, D))

        predicted_xs = torch.cat([predicted_x0, predicted_xs], axis=2)

        loss = self.observation_likelihood(predicted_xs, ys.unsqueeze(0))

        shooting_loss = self.constraint_likelihood(
            ss_samples[:, :, 1:, :],
            predicted_xs[:, :, 0:-shooting_time_factor:shooting_time_factor, :],
        )
        return loss, self.alpha * shooting_loss


class SequenceModel(BaseSequenceModel):
    """
    Standard ODE model, with no shooting, works for irregular timepoints.
    """

    def __init__(
        self,
        flow,
        num_observations,
        state_distribution,
        observation_likelihood,
        constraint_likelihood,
        ts_dense_scale=2,
    ):
        super(SequenceModel, self).__init__(
            flow=flow,
            num_observations=num_observations,
            state_distribution=state_distribution,
            observation_likelihood=observation_likelihood,
            constraint_likelihood=constraint_likelihood,
            ts_dense_scale=ts_dense_scale,
        )

    def build_objective(self, ys, ts):
        """
        Compute objective.
        @param ys: true observation sequence
        @param ts: observation timesd
        @return: loss, nll, initial_staet_kl, inducing_kl
        """
        observ_loglik, init_state_kl = self.build_lowerbound_terms(ys, ts)
        kl = self.build_inducing_kl()
        loss = -(observ_loglik - init_state_kl - kl)
        return loss

    def build_lowerbound_terms(self, ys, ts, num_samples=1):
        """
        Given oberved states and time, builds the individual terms for the lowerbound computation

        @param ys: observed sequence tensor (N,T,D)
        @param ts: observed time sequence (T,)
        @return: nll, initial state KL
        """
        assert num_samples == 1, ">1 sample not implemented for standard model."
        ts = insert_zero_t0(ts)
        x0_samples = self.state_distribution.sample(num_samples=1)[0]
        x0_kl = self.state_distribution.kl()
        xs = self.build_flow(x0_samples, ts)[:, 1:]
        loglik = self.observation_likelihood.log_prob(xs, ys)
        return loglik.mean(), x0_kl.mean() / self.num_observations


class SubSequenceModel(BaseSequenceModel):
    """
    Batched data model, with timepoints on regular grid.
    """

    def __init__(
        self,
        flow,
        num_observations,
        state_distribution,
        observation_likelihood,
        constraint_likelihood,
        ts_dense_scale=2,
    ):
        super(SubSequenceModel, self).__init__(
            flow=flow,
            num_observations=num_observations,
            state_distribution=state_distribution,
            observation_likelihood=observation_likelihood,
            constraint_likelihood=constraint_likelihood,
            ts_dense_scale=ts_dense_scale,
        )

    def build_objective(self, ys, ts):
        """
        Compute objective for GPODE optimization
        @param ys: true observation sequence
        @param ts: observation timesd
        @return: loss, nll, initial_staet_kl, inducing_kl
        """
        observ_loglik = self.build_lowerbound_terms(ys, ts)
        kl = self.build_inducing_kl()
        loss = -(observ_loglik - kl)
        return loss

    def build_lowerbound_terms(self, ys_batched, ts, num_samples=1):
        """
        Given oberved states and time, builds the individual terms for the lowerbound computation

        @param ys: observed sequence tensor (N_batch,T,D)
        @param ts: observed time sequence (T,)
        @return: nll, initial state KL
        """
        assert num_samples == 1, ">1 sample not implemented for standard model."
        xs = self.build_flow(ys_batched[:, 0, :], ts)
        loglik = self.observation_likelihood.log_prob(xs, ys_batched)
        return loglik.mean()


class UniformShootingModel(BaseSequenceModel):
    """
    Implements shooting model for data observed on uniform time grid.

    Defines following methods:
        build_lowerbound_terms: given observed states and time, builds individual terms for the lowerbound computation
    """

    def __init__(
        self,
        flow,
        num_observations,
        state_distribution,
        observation_likelihood,
        constraint_likelihood,
        shooting_time_factor=None,
        energy_likelihood=None,
        ts_dense_scale=2,
    ):
        super(UniformShootingModel, self).__init__(
            flow=flow,
            num_observations=num_observations,
            state_distribution=state_distribution,
            observation_likelihood=observation_likelihood,
            constraint_likelihood=constraint_likelihood,
            ts_dense_scale=ts_dense_scale,
        )

        self.energy_likelihood = energy_likelihood
        self.constrain_energy = False
        self.shooting_time_factor = shooting_time_factor

    def compute_segments(self, ts, num_samples=1, constrain_energy=False):

        ss_samples = self.state_distribution.sample(
            num_samples=num_samples
        )  # (S,N,(T-1)/shooting_time_factor + 1, D)
        (S, N, N_state, D) = ss_samples.shape
        N_shooting = N_state - 1
        T = ts.shape[0]
        shooting_time_factor = (T - 1) // N_shooting

        assert (
            shooting_time_factor == self.shooting_time_factor
        ), f"{shooting_time_factor}, {T}, {N_shooting}"

        if constrain_energy:

            predicted_xs, predicted_energy = self.flow(
                x0=stack_segments(ss_samples),
                ts=ts[: shooting_time_factor + 1],
                return_energy=True,
            )  # (SxNxN_state, shooting_time_factor+1, D)

            # get the energy of the initial point of each segement
            # include initial x0 in ss energy as it also needs to be penalised
            ss_energy = unstack_segments(predicted_energy[:, 0], (S, N, N_state, 1))
        else:
            predicted_xs = self.flow(
                x0=stack_segments(ss_samples),
                ts=ts[: shooting_time_factor + 1],
                return_energy=False,
            )  # (SxNxN_state, shooting_time_factor+1, D)

        # get additional set of points we don't need from integrating the initial
        # condition too far,
        predicted_xs = unstack_segments(
            predicted_xs[:, 1:], (S, N, T + shooting_time_factor - 1, D)
        )
        # get rid of extraneous intial points
        predicted_xs = torch.cat(
            [
                predicted_xs[:, :, 0:1, :],
                predicted_xs[:, :, shooting_time_factor:, :],
            ],
            axis=2,
        )

        if constrain_energy:
            return ss_samples, predicted_xs, ss_energy
        else:
            return ss_samples, predicted_xs

    def build_lowerbound_terms(self, ys, ts, num_samples=1):
        """
        Given observed states and time, builds the individual terms for the lowerbound computation

        @param ys: observed sequence tensor (N,T,D)
        @param ts: observed time sequence (T,)
        @param num_samples: number of reparametrized samples used to compute lowerbound
        @return: nll, state cross-entropy, state entropy, initial state KL
        """

        ss_samples, predicted_xs = self.compute_segments(
            ts, num_samples=num_samples, constrain_energy=False
        )

        (S, N, N_state, D) = ss_samples.shape
        N_shooting = N_state - 1
        T = ts.shape[0]
        shooting_time_factor = (T - 1) // N_shooting

        assert (
            shooting_time_factor == self.shooting_time_factor
        ), f"{shooting_time_factor}, {T}, {N_shooting}"

        observation_loglik = self.observation_likelihood.log_prob(
            predicted_xs, ys.unsqueeze(0)
        )  # (S,N,T,D)

        # compute the entropy of variational posteriors for shooting states
        state_entropy = self.state_distribution.entropy()  # (N,T-1)

        # compute the shooting constraint likelihoods
        state_constraint_logprob = self.constraint_likelihood.log_prob(
            ss_samples[:, :, 1:, :],
            predicted_xs[:, :, 0:-shooting_time_factor:shooting_time_factor, :],
        ).sum(
            3
        )  # (S,N,T-1)

        # compute initial state KL
        initial_state_kl = self.state_distribution.x0.kl()  # (1,)

        assert state_entropy.shape == (N, N_shooting)
        assert state_constraint_logprob.shape == (S, N, N_shooting)

        total_state_constraint_loglik = (
            state_constraint_logprob.mean(0).sum()
        ) / self.num_observations

        scaled_state_entropy = state_entropy.sum() / self.num_observations
        scaled_initial_state_kl = initial_state_kl / self.num_observations
        return (
            observation_loglik.mean(),
            total_state_constraint_loglik,
            scaled_state_entropy,
            scaled_initial_state_kl,
        )


class ConsUniformShootingModel(UniformShootingModel):
    """
    Energy conserving shooting model, with timepoints on regular grid.
    """

    def __init__(
        self,
        flow,
        num_observations,
        state_distribution,
        observation_likelihood,
        constraint_likelihood,
        shooting_time_factor=None,
        energy_likelihood=None,
        ts_dense_scale=2,
    ):
        super(ConsUniformShootingModel, self).__init__(
            flow=flow,
            num_observations=num_observations,
            state_distribution=state_distribution,
            observation_likelihood=observation_likelihood,
            constraint_likelihood=constraint_likelihood,
            shooting_time_factor=shooting_time_factor,
            energy_likelihood=energy_likelihood,
            ts_dense_scale=ts_dense_scale,
        )

        self.constrain_energy = True

    def build_lowerbound_terms(self, ys, ts, num_samples=1):
        """
        Given observed states and time, builds the individual terms for the lowerbound computation

        @param ys: observed sequence tensor (N,T,D)
        @param ts: observed time sequence (T,)
        @param num_samples: number of reparametrized samples used to compute lowerbound
        @return: nll, state cross-entropy, state entropy, initial state KL
        """

        ss_samples, predicted_xs, ss_energy = self.compute_segments(
            ts, num_samples=num_samples, constrain_energy=True
        )

        (S, N, N_state, D) = ss_samples.shape
        N_shooting = N_state - 1
        T = ts.shape[0]
        shooting_time_factor = (T - 1) // N_shooting

        assert (
            shooting_time_factor == self.shooting_time_factor
        ), f"{shooting_time_factor}, {T}, {N_shooting}"

        observation_loglik = self.observation_likelihood.log_prob(
            predicted_xs, ys.unsqueeze(0)
        )  # (S,N,T,D)

        # compute the entropy of variational posteriors for shooting states
        state_entropy = self.state_distribution.entropy()  # (N,T-1)

        # compute the shooting constraint likelihoods
        state_constraint_logprob = self.constraint_likelihood.log_prob(
            ss_samples[:, :, 1:, :],
            predicted_xs[:, :, 0:-shooting_time_factor:shooting_time_factor, :],
        ).sum(
            3
        )  # (S,N,T-1)

        # compute initial state KL
        initial_state_kl = self.state_distribution.x0.kl()  # (1,)

        assert state_entropy.shape == (N, N_shooting)
        assert state_constraint_logprob.shape == (S, N, N_shooting)

        scaled_state_constraint_loglik = (
            state_constraint_logprob.mean(0).sum()
        ) / self.num_observations

        # compute the energy likelihood
        energy_constraint_logprob = self.energy_likelihood.log_prob(
            ss_energy[:, :, 1:-1, :], ss_energy[:, :, 2:, :]
        ).squeeze(3)

        # print(energy_constraint_logprob[:, :, 0:3])

        scaled_energy_constraint_loglik = (
            energy_constraint_logprob.mean(0).sum() / self.num_observations
        )

        # print(energy_constraint_logprob.mean(0).sum() / self.num_observations)
        scaled_state_entropy = state_entropy.sum() / self.num_observations
        scaled_initial_state_kl = initial_state_kl / self.num_observations
        return (
            observation_loglik.mean(),
            scaled_state_constraint_loglik,
            scaled_energy_constraint_loglik,
            scaled_state_entropy,
            scaled_initial_state_kl,
        )
