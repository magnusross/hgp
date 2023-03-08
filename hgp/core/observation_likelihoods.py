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

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init

from hgp.misc.constraint_utils import invsoftplus, softplus


class Gaussian(nn.Module):
    """
    Gaussian likelihood
    """

    def __init__(self, ndim=1, init_val=0.01):
        super(Gaussian, self).__init__()
        self.unconstrained_variance = torch.nn.Parameter(
            torch.ones(ndim), requires_grad=True
        )
        self._initialize(init_val)

    def _initialize(self, x):
        init.constant_(self.unconstrained_variance, invsoftplus(torch.tensor(x)).item())

    @property
    def variance(self):
        return softplus(self.unconstrained_variance)

    @variance.setter
    def variance(self, value):
        self.unconstrained_variance = nn.Parameter(
            invsoftplus(value), requires_grad=True
        )

    def log_prob(self, F, Y):
        return -0.5 * (
            np.log(2.0 * np.pi)
            + torch.log(self.variance)
            + torch.pow(F - Y, 2) / self.variance
        )
