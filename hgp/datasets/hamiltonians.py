# import numpy as np
import functorch
import numpy as np
import torch
from torchdiffeq import odeint

from hgp.misc.ham_utils import build_J
from hgp.misc.torch_utils import numpy2torch, torch2numpy

# from scipy.integrate import odeint


class Data:
    def __init__(self, ys, ts):
        self.ts = ts.astype(np.float32)
        self.ys = ys.astype(np.float32)

    def __len__(self):
        return self.ys.shape[0]

    def __getitem__(self, index):
        return self.ys[index, ...], self.ts


class HamiltonianSystem:
    def __init__(
        self,
        state_dimension,
        frequency_train=4,
        T_train=6.0,
        frequency_test=None,
        T_test=None,
        x0=None,
        x0_test=None,
        N_x0s=None,
        N_x0s_test=None,
        noise_var=0.01,
        noise_rel=False,
        device="cpu",
        seed=121,
        ic_mode=None,
    ):
        noise_rng = np.random.RandomState(seed)
        init_rng_train = np.random.RandomState(seed + 1)
        init_rng_test = np.random.RandomState(seed + 2)

        frequency_test = (
            frequency_test if frequency_test is not None else frequency_train
        )
        T_test = T_test if T_test is not None else T_train

        self.S_test = frequency_train
        self.T_test = T_test
        self.S_train = frequency_test
        self.T_train = T_train

        if N_x0s is None:
            N_x0s = 10

        if x0 is None:
            x0 = self.sample_ics(N_x0s, ic_mode=ic_mode, rng=init_rng_train)

        if x0_test is None and N_x0s_test is not None:
            x0_test = self.sample_ics(N_x0s_test, ic_mode=ic_mode, rng=init_rng_test)

        if N_x0s_test is None:
            N_x0s_test = N_x0s

        if x0_test is None:
            x0_test = x0

        self.state_dimension = state_dimension
        self.J = build_J(state_dimension).float()
        self.x0 = x0
        self.x0_test = x0_test
        self.noise_var = noise_var

        xs_train, ts_train = self.generate_sequence(
            x0=self.x0, sequence_length=int(frequency_train * T_train) + 1, T=T_train
        )
        xs_test, ts_test = self.generate_sequence(
            x0=self.x0_test, sequence_length=int(frequency_test * T_test), T=T_test
        )
        xs_train = xs_train + noise_rng.normal(size=xs_train.shape) * (
            self.noise_var**0.5
        ) * (1.0 if not noise_rel else xs_train.std(axis=(1))[:, None, :])

        self.trn = Data(ys=xs_train, ts=ts_train)
        self.tst = Data(ys=xs_test, ts=ts_test)

        self.mean_std_ys = self.trn.ys.mean(axis=(0, 1)), self.trn.ys.std(axis=(0, 1))
        self.max_trn = self.trn.ts.max()

    def f(self, t, x):
        """
        Computes derivative function from H
        """
        dHdx = functorch.grad(lambda xi: self.hamiltonian(xi).sum())(x)
        return dHdx @ self.J.T

    def generate_sequence(self, x0, sequence_length, T):
        """
        Generates trajectories given derivative function
        """
        with torch.no_grad():
            ts = torch.linspace(0, 1, sequence_length) * T
            # x0 = torch.tensor(x0, dtype=torch.float32, requires_grad=False)
            x0 = x0.clone().detach()
            xs = torch2numpy(
                odeint(
                    self.f,
                    x0,
                    ts,
                ).permute(1, 0, 2)
            )
        return xs, torch2numpy(ts)

    def generate_purturbed_ensemble(
        self, x0, sequence_length, T, dx=0.01, N=2, scale=True
    ):
        xs = []
        if scale:
            x0 = self.unscale_output(x0)

        with torch.no_grad():
            for n in range(N):
                dx0 = np.random.uniform(low=-dx, high=dx, size=x0.shape)
                x0_per = torch.tensor(
                    x0 + dx0, dtype=torch.float32, requires_grad=False
                )
                xsi, _ = self.generate_sequence(x0_per, sequence_length, T)
                xs.append(xsi)

        out = np.stack(xs)
        if scale:
            out = self.scale_output(out)

        return out

    def scale_output(self, x):
        return (x - self.mean_std_ys[0]) / self.mean_std_ys[1]

    def unscale_output(self, x):
        return x * self.mean_std_ys[1] + self.mean_std_ys[0]

    def scale_t(self, t):
        return t / self.max_trn

    def unscale_t(self, t):
        return t * self.max_trn

    def scale_ts(self):
        self.trn.ts = self.scale_t(self.trn.ts)
        self.tst.ts = self.scale_t(self.tst.ts)

    def scale_ys(self):
        self.tst.ys = self.scale_output(self.tst.ys)
        self.trn.ys = self.scale_output(self.trn.ys)

        self.x0_test = self.scale_output(self.x0_test)
        self.x0 = self.scale_output(self.x0)


class SimplePendulum(HamiltonianSystem):
    def __init__(
        self,
        **kwargs,
    ):
        super(SimplePendulum, self).__init__(2, **kwargs)

        self.xlim = (
            -self.tst.ys[:, :, 0].max() - 0.1,
            self.tst.ys[:, :, 0].max() + 0.1,
        )
        self.ylim = (
            -self.tst.ys[:, :, 1].max() - 0.1,
            self.tst.ys[:, :, 1].max() + 0.1,
        )
        self.name = "simple-pendulum"

    def sample_ics(self, N, rng, ic_mode=None):
        # if ic_mode == "greydanus" or ic_mode is None:
        out = []
        n = 0

        while n < N:
            x0 = rng.rand(2) * 2.0 - 1.0
            energy = self.hamiltonian(numpy2torch(x0))
            if energy < 9.81:
                out.append(x0)
                n += 1
        return torch.tensor(np.array(out), dtype=torch.float32)

    def hamiltonian(self, x, m=1, g=9.81, r=1):
        q, p = torch.split(x, x.shape[-1] // 2, dim=-1)
        return m * g * r * (1 - torch.cos(q)) + 0.5 / (r**2 * m) * p**2


class SpringPendulum(HamiltonianSystem):
    def __init__(
        self,
        **kwargs,
    ):

        self.m = 1.0
        self.l0 = 3.0
        self.k = 10
        self.g = 9.81

        super(SpringPendulum, self).__init__(4, **kwargs)

        self.lim = self.trn.ys.max()
        self.name = "spring-pendulum"

    def sample_ics(self, N, rng, ic_mode=None):

        out_ics = torch.tensor(
            rng.uniform(low=-0.25, high=0.25, size=(N, 4)), dtype=torch.float32
        )
        return out_ics

    def hamiltonian(self, x):
        q, p = torch.split(x, x.shape[-1] // 2, dim=-1)
        kin = (
            0.5
            * (1 / self.m)
            * (p[..., 0] ** 2 + p[..., 1] ** 2 / (q[..., 0] + self.l0) ** 2)
        )
        elas = 0.5 * self.k * (q[..., 0]) ** 2
        gpe = -self.m * self.g * (q[..., 0] + self.l0) * torch.cos(q[..., 1])
        return kin + elas + gpe


class HenonHeiles(HamiltonianSystem):
    def __init__(
        self,
        **kwargs,
    ):
        self.mu = 0.8
        super(HenonHeiles, self).__init__(4, **kwargs)

        self.lim = self.trn.ys.max()
        self.name = "henon-heiles"

    def sample_ics(self, N, rng, ic_mode=None):
        out = []
        n = 0
        while n < N:

            x0 = rng.uniform(low=-1.0, high=1.0, size=4)

            energy = self.hamiltonian(numpy2torch(x0))
            if energy < 1 / (6 * self.mu**2):
                out.append(x0)
                n += 1
        out_ics = torch.tensor(np.array(out), dtype=torch.float32)
        return out_ics

    def hamiltonian(self, x):
        q, p = torch.split(x, x.shape[-1] // 2, dim=-1)
        return self.mu * (q[..., 0] ** 2 * q[..., 1] - q[..., 1] ** 3 / 3) + 0.5 * (
            x**2
        ).sum(-1)


def load_system_from_name(name):
    all_classes = {
        "simple-pendulum": SimplePendulum,
        "henon-heiles": HenonHeiles,
        "spring-pendulum": SpringPendulum,
    }
    return all_classes[name]
