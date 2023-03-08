import pytest

import hgp.core.kernels as kernels
import torch


@pytest.fixture()
def t():
    return 1 * torch.randn(10, 4)


@pytest.fixture()
def kernel():
    k = kernels.DerivativeRBF(4)
    return k


def test_single_k(t, kernel):
    K = kernel.K(t)
    for i in range(10):
        for j in range(10):
            assert torch.isclose(K[i, j], kernel.single_k(t[i], t[j]))


def test_grad_single_k(t, kernel):
    def analytic_dk(xi, yi, dim):
        return (
            -0.5
            * (1 / kernel.lengthscales[dim] ** 2)
            * (xi[dim] - yi[dim])
            * kernel.single_k(xi, yi)
        )

    for j in range(10):
        for k in range(10):
            for i in range(4):
                print(i)
                assert torch.isclose(
                    kernel.grad_single_k(t[k], t[j])[i],
                    analytic_dk(t[k], t[j], i),
                )


def test_hess_single_k(t, kernel):
    def analytic_ddk(xi, yi, dim1, dim2):
        return (
            0.25
            * (1 / kernel.lengthscales[dim2] ** 2)
            * (
                2 * (dim1 == dim2)
                - (1 / kernel.lengthscales[dim1] ** 2)
                * (xi[dim1] - yi[dim1])
                * (xi[dim2] - yi[dim2])
            )
            * kernel.single_k(xi, yi)
        )

    for j in range(10):
        for k in range(10):
            for i in range(4):
                for j in range(4):
                    pred = kernel.hess_single_k(t[j], t[k])[i][j]
                    ana = analytic_ddk(t[j], t[k], i, j)
                    print(pred, ana)
                    assert torch.isclose(
                        pred,
                        ana,
                    )


def test_grad_K(t, kernel):
    dK = kernel.grad_K(t, t[:9])

    def analytic_dk(xi, yi, dim):
        return (
            -0.5
            * (1 / kernel.lengthscales[dim] ** 2)
            * (xi[dim] - yi[dim])
            * kernel.single_k(xi, yi)
        )

    # print(dK.shape)

    for i in range(4 * 10):
        for j in range(9):
            assert torch.isclose(
                dK[i, j], analytic_dk(t[i % 10], t[j], i // 10), atol=1e-6
            )


def test_hess_K(t, kernel):
    ddK = kernel.hess_K(t)

    def analytic_ddk(xi, yi, dim1, dim2):
        return (
            0.25
            * (1 / kernel.lengthscales[dim2] ** 2)
            * (
                2 * (dim1 == dim2)
                - (1 / kernel.lengthscales[dim1] ** 2)
                * (xi[dim1] - yi[dim1])
                * (xi[dim2] - yi[dim2])
            )
            * kernel.single_k(xi, yi)
        )

    for i in range(4 * 10):
        for j in range(4 * 10):
            assert torch.isclose(
                ddK[i, j],
                analytic_ddk(t[i % 10], t[j % 10], i // 10, j // 10),
                atol=1e-6,
            )


def test_hess_K_PSD(t, kernel):
    ddK = kernel.hess_K(t)
    torch.linalg.cholesky(ddK + torch.eye(ddK.shape[1]) * 1e-5)


def test_variance_setter(kernel):
    kernel.variance = torch.tensor(2.70)
    assert torch.isclose(kernel.variance, torch.tensor(2.70))


def test_lengthscale_setter(kernel):
    kernel.lengthscales = torch.ones(4) * 2.7
    assert torch.all(torch.isclose(kernel.lengthscales, torch.ones(4) * 2.7))
