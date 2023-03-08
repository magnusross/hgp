import functorch
import torch
from torch import nn

from hgp.misc.ham_utils import build_J


def Linear(chin, chout, zero_bias=False, orthogonal_init=False):
    linear = nn.Linear(chin, chout)
    if zero_bias:
        torch.nn.init.zeros_(linear.bias)
    if orthogonal_init:
        torch.nn.init.orthogonal_(linear.weight, gain=0.5)
    return linear


def FCtanh(chin, chout, zero_bias=False, orthogonal_init=False):
    return nn.Sequential(Linear(chin, chout, zero_bias, orthogonal_init), nn.Tanh())


class NNModel(nn.Module):
    def __init__(self, D_in, D_out, N_nodes, N_layers):
        super(NNModel, self).__init__()
        self.D_out = D_out
        self.D_in = D_in

        chs = [self.D_in] + N_layers * [N_nodes]
        self.net = nn.Sequential(
            *[
                FCtanh(chs[i], chs[i + 1], zero_bias=False, orthogonal_init=False)
                for i in range(N_layers)
            ],
            Linear(chs[-1], D_out, zero_bias=False, orthogonal_init=False)
        )

    def forward(self, t, x):
        return self.net(x)

    def build_cache(self):
        pass


class HamiltonianNNModel(NNModel):
    """
    Implements a NN model with Hamiltonian restriction.
    """

    def __init__(self, D_in, N_nodes, N_layers):
        super(HamiltonianNNModel, self).__init__(D_in, 1, N_nodes, N_layers)

        self.J = build_J(D_in)

    def hamiltonian(self, t, x):
        H = super(HamiltonianNNModel, self).forward(t, x)
        return H[:, 0]

    def forward(self, t, x):
        dHdx = functorch.grad(lambda xi: self.hamiltonian(t, xi).sum())(x)
        f = dHdx @ self.J.T
        return f
