from hgp.misc.torch_utils import torch2numpy, numpy2torch

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import shutil
import numpy as np

from hgp.models.builder import compute_summary


if shutil.which("latex"):
    matplotlib.rc("text", usetex=True)
    matplotlib.rc("text.latex", preamble=r"\usepackage[cm]{sfmath}")
    font = {
        "size": 10
    }
    matplotlib.rc("font", **font)
else:
    print("LaTeX not installed, using default backend.")


def plot_predictions(data, test_pred, save=None, test_true=None, model_name="Model"):

    test_ts, test_ys = data.tst.ts, data.tst.ys

    fig, axs = plt.subplots(
        data.state_dimension,
        1,
        figsize=(
            12,
            2 * data.state_dimension,
        ),
        sharex=True,
        sharey=True,
        squeeze=True,
    )
    # for i, (model_name, test_pred) in enumerate(test_pred_dict.items()):
    for d in range(data.state_dimension):
        # axs[d].plot(test_ts, test_pred_mean[n, :, d], c="r", alpha=0.7, zorder=3)
        axs[d].plot(test_ts, test_pred[:, 0, :, d].T, c=cm.Set1(2), alpha=0.1, zorder=4)
        axs[d].plot(test_ts, test_ys[0, :, d], c="k", alpha=0.7, zorder=2)
        axs[d].scatter(
            data.trn.ts,
            data.trn.ys[0, :, d],
            c="k",
            s=20,
            marker=".",
            zorder=200,
        )
        axs[d].set_xlabel("$t$")
        axs[d].scatter([], [], c="k", s=20, marker=".", label="Training data")
        axs[d].plot([], [], c="k", alpha=0.7, label="True $\mathbf{x}(t)$")
        axs[d].plot(
            [],
            [],
            c=cm.Set1(2),
            alpha=0.9,
            label=model_name + " Pred. $\mathbf{x}(t)$",
        )

    axs[0].legend(loc="upper right")

    axs[0].set_title(model_name)

    for i in range(data.state_dimension):
        half_d = data.state_dimension // 2
        ax_label = ("q_" if i < half_d else "p_") + str(i % half_d + 1)
        axs[i].set_ylabel(f"${ax_label}$")

    # fig.suptitle("Predictive posterior")
    # fig.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.tight_layout()
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()


def plot_longitudinal(data, test_pred, noisevar, save=None, test_true=None):
    test_pred_mean, test_pred_postvar = test_pred.mean(0), test_pred.var(0)
    test_pred_predvar = test_pred_postvar + noisevar

    if test_true is None:
        test_ts, test_ys = data.tst.ts, data.tst.ys
    else:
        test_ts, test_ys = test_true

    for n in range(test_pred_mean.shape[0]):
        fig, axs = plt.subplots(
            data.state_dimension,
            1,
            figsize=(
                3 * data.state_dimension,
                8 * 1,
            ),
        )
        for d in range(data.state_dimension):
            axs[d].plot(test_ts, test_pred_mean[n, :, d], c="r", alpha=0.7, zorder=3)
            axs[d].fill_between(
                test_ts,
                test_pred_mean[n, :, d] - 2 * test_pred_postvar[n, :, d] ** 0.5,
                test_pred_mean[n, :, d] + 2 * test_pred_postvar[n, :, d] ** 0.5,
                color="r",
                alpha=0.1,
                zorder=1,
                label="posterior",
            )
            axs[d].fill_between(
                test_ts,
                test_pred_mean[n, :, d] - 2 * test_pred_predvar[n, :, d] ** 0.5,
                test_pred_mean[n, :, d] + 2 * test_pred_predvar[n, :, d] ** 0.5,
                color="b",
                alpha=0.1,
                zorder=0,
                label="predictive",
            )
            axs[d].plot(test_ts, test_pred[:, n, :, d].T, c="g", alpha=0.1, zorder=4)
            axs[d].plot(test_ts, test_ys[n, :, d], c="k", alpha=0.7, zorder=2)
            if data.trn.ys.shape[0] == data.tst.ys.shape[0]:
                axs[d].scatter(
                    data.trn.ts,
                    data.trn.ys[n, :, d],
                    c="k",
                    s=100,
                    marker=".",
                    zorder=200,
                )
            axs[d].set_title("State {}".format(d + 1))
            axs[d].set_xlabel("Time")
            axs[d].scatter([], [], c="k", s=10, marker=".", label="train obs")
            axs[d].plot([], [], c="k", alpha=0.7, label="true")
            axs[d].plot([], [], c="r", alpha=0.7, label="predicted")
        axs[d].legend(loc="upper right")
        fig.suptitle("Predictive posterior")
        fig.subplots_adjust(wspace=0.2, hspace=0.2)
        plt.tight_layout()
        if save:
            plt.savefig(save + f"t{n}.pdf")
            plt.close()
        else:
            return fig, axs


def plot_traces(model, data, test_pred, save=None):
    mll, mse = compute_summary(
        data.tst.ys,
        torch2numpy(test_pred),
        torch2numpy(model.observation_likelihood.variance),
        squeeze_time=False,
    )
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    (ax1, ax2, ax3, ax4) = axs.flatten()
    ax1.plot(data.tst.ts, mll.T)
    ax1.set_title("MLL")
    ax2.plot(data.tst.ts, mse.T)
    ax2.set_title("MSE")
    ax3.plot(data.tst.ts, np.var(torch2numpy(test_pred), axis=0).mean(-1).T)
    ax3.set_title("Variance")

    pred_energy = torch2numpy(data.hamiltonian(numpy2torch(test_pred)))
    true_energy = torch2numpy(data.hamiltonian(numpy2torch(data.tst.ys)))
    energy_err = np.power(true_energy - pred_energy.mean(0), 2)

    ax4.plot(data.tst.ts, energy_err.T)
    ax4.set_title("Energy MSE")

    if save:
        plt.savefig(save)
    else:
        plt.show()


def plot_comparison_traces(test_preds, obs_noises, data, save=None, names=None):
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    names = [None, None] if names is None else names
    for i, (noise, test_pred, name) in enumerate(zip(obs_noises, test_preds, names)):
        mll, mse, rel_err = compute_summary(
            data.tst.ys,
            torch2numpy(test_pred),
            torch2numpy(noise),
            squeeze_time=False,
        )

        (ax1, ax2, ax3, ax4) = axs.flatten()
        ax1.plot(data.tst.ts, mll.T, c=cm.Set2(i), alpha=0.7)
        ax1.plot([], [], c=cm.Set2(i), label=name)

        ax1.set_title("MLL")

        ax2.plot(data.tst.ts, rel_err.T, c=cm.Set2(i), alpha=0.7)
        ax2.set_title("Relative Error")
        ax3.plot(
            data.tst.ts,
            np.var(torch2numpy(test_pred), axis=0).mean(-1).T,
            c=cm.Set2(i),
            alpha=0.7,
        )
        ax3.set_title("Variance")

        pred_energy = torch2numpy(data.hamiltonian(numpy2torch(test_pred)))
        true_energy = torch2numpy(data.hamiltonian(numpy2torch(data.tst.ys)))
        energy_err = np.sqrt(np.power(true_energy - pred_energy.mean(0), 2))

        ax4.plot(data.tst.ts, np.squeeze(energy_err).T, c=cm.Set2(i), alpha=0.7)
        ax4.set_title("Energy MSE")

    for ax in axs.flatten():
        ax.axvline(
            data.trn.ts.max(),
            ls=":",
            c="grey",
            alpha=0.5,
            label="End of train period",
        )
        ax.set_xlabel("T (s)")
    ax1.legend()

    if save:
        plt.savefig(save)
    else:
        plt.show()


def plot_learning_curve(history, save=None):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 3))

    ax1.plot(history.loss_meter.iters, history.loss_meter.vals)
    ax1.set_title("Loss function")
    ax1.set_yscale("log")
    try:
        ax2.plot(
            history.observation_nll_meter.iters, history.observation_nll_meter.vals
        )
        ax2.set_title("Observation NLL")
        # ax2.set_yscale("log")
        ax3.plot(history.state_kl_meter.iters, history.state_kl_meter.vals)
        ax3.set_title("State KL")
        ax3.set_yscale("log")
    # deals with nn plotting
    except AttributeError:
        pass

    if save:
        plt.savefig(save)
    else:
        plt.show()
