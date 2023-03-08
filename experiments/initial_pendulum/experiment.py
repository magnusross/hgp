import logging
import os

import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import shutil
import torch
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.legend import _get_legend_handles_labels
from omegaconf import DictConfig

import hgp
from hgp.datasets.hamiltonians import load_system_from_name
from hgp.misc.torch_utils import numpy2torch, torch2numpy
from hgp.misc.train_utils import seed_everything

log = logging.getLogger(__name__)


@hydra.main(config_path=".", config_name="config", version_base="1.2")
def run_experiment(config: DictConfig):
    print("Working directory : {}".format(os.getcwd()))

    seed_everything(3)
    (fig, axs) = plt.subplots(
        2,
        4,
        figsize=(12, 6),
        # sharex=True,
        # sharey=True,
    )

    cycle_lengths = [0.52, 1]
    model_names = ["hgp", "gpode"]

    titles1 = ["Inferred vector field", "Posterior samples"]

    titles2 = [", $\\frac{1}{2}$ cycle", ", $1$ cycle"]

    for c, cycle_length in enumerate(cycle_lengths):
        train_time = cycle_length * 2 * np.pi / np.sqrt(9.81)
        system = load_system_from_name("simple-pendulum")(
            frequency_train=16,
            T_train=train_time,
            frequency_test=20,
            T_test=(8 * np.pi / np.sqrt(9.81)),
            noise_var=0.01,
            noise_rel=True,
            seed=3,
            N_x0s=1,
        )

        models = []
        preds = []
        for model_name in model_names:
            model = (
                hgp.models.builder.build_model(config, system.trn.ys)
                if model_name == "hgp"
                else hgp.models.builder.build_gpode_model(config, system.trn.ys)
            )
            model = hgp.models.builder.init_and_fit(
                model, config, system.trn.ts, system.trn.ys
            )
            models.append(model)
            preds.append(
                hgp.models.builder.compute_predictions(
                    model,
                    numpy2torch(system.tst.ts),
                    eval_sample_size=config.eval_samples,
                )
            )

        (ax1, ax2, ax3, ax4) = axs[c]

        grid_size = 30
        xlim = system.xlim
        ylim = system.ylim
        factor = 1.5
        xx, yy = np.meshgrid(
            np.linspace(xlim[0] * factor, xlim[1] * factor, grid_size),
            np.linspace(ylim[0] * factor, ylim[1] * factor, grid_size),
        )

        grid_x = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], 1)
        grid_f = []
        for gx in grid_x:
            grid_f.append(
                torch2numpy(system.f(None, torch.tensor(gx, dtype=torch.float32)))
            )
        grid_f = np.stack(grid_f)

        if c == 0:
            ax1.streamplot(
                xx,
                yy,
                grid_f[:, 0].reshape(xx.shape),
                grid_f[:, 1].reshape(xx.shape),
                color="grey",
                density=0.5,
            )
            ax1.set_title("True vector field")
            ax1.scatter(
                [None], [None], marker=".", c="k", alpha=0.8, label="Training data"
            )
            ax1.plot(
                [None],
                [None],
                color="k",
                linestyle="solid",
                alpha=1.0,
                zorder=4,
                label="True trajectory",
            )
            ax1.set_ylabel("$p$")
            ax1.set_xlabel("$q$")

            # ax1.legend(loc="lower right")
        else:
            ax1.axis("off")

        grid_x = torch.tensor(
            np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], 1),
            dtype=torch.float32,
        )
        for i, model in enumerate(models):
            grid_f = []
            color = cm.Set2(i)
            with torch.no_grad():
                for _ in range(100):
                    model.flow.odefunc.diffeq.build_cache()
                    grid_f.append(model.flow.odefunc.diffeq.forward(None, grid_x))
            grid_f = torch2numpy(torch.stack(grid_f))
            sp = ax2.streamplot(
                xx,
                yy,
                grid_f.mean(0)[:, 0].reshape(xx.shape),
                grid_f.mean(0)[:, 1].reshape(xx.shape),
                color=color,
                arrowsize=1.1,
                arrowstyle="<|-",
                density=0.5,
            )

            ax2.set_title(titles1[0] + titles2[c])

            if c == 0:
                ax2.plot(
                    [None],
                    [None],
                    linestyle=":" if i == 0 else "solid",
                    color=color,
                    label=model_names[i].upper(),
                )

            sp.lines.set(alpha=0.8, ls=":" if i == 0 else "solid")

            for s in range(min(preds[i].shape[0], 10)):
                for n in range(preds[i].shape[1]):
                    points = preds[i][s, n].reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    lc = LineCollection(
                        segments,
                        linestyle=":" if i == 0 else "solid",
                        alpha=0.3,
                        color=color,
                    )
                    lc.set_linewidth(2.5)
                    ax3.add_collection(lc)

            ax3.set_title(titles1[1] + titles2[c])

            pred_energy = torch2numpy(system.hamiltonian(numpy2torch(preds[i])))
            true_energy = torch2numpy(system.hamiltonian(numpy2torch(system.tst.ys)))
            energy_err = np.sqrt(np.power(true_energy - pred_energy, 2))
            ax4.plot(
                system.tst.ts,
                np.squeeze(energy_err).T,
                linestyle=":" if i == 0 else "solid",
                alpha=0.3,
                color=color,
            )
            ax4.set_title("Energy MSE" + titles2[c])
            ax4.set_xlabel("$t$")

        for n in range(system.tst.ys.shape[0]):
            points = system.tst.ys[n].reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(
                segments, color="k", linestyle="solid", alpha=1.0, zorder=4
            )
            lc.set_linewidth(0.5)
            ax3.add_collection(lc)

        ax3.scatter(
            system.trn.ys[:, :, 0], system.trn.ys[:, :, 1], marker=".", c="k", alpha=1
        )

        for ax in [ax1, ax2, ax3]:
            ax.set_xlim(xlim[0] * factor, xlim[1] * factor)
            ax.set_ylim(ylim[0] * factor, ylim[1] * factor)

    fig.legend(*_get_legend_handles_labels(fig.axes), loc=(0.1, 0.3))
    plt.tight_layout()
    plt.savefig("./figure4.pdf")


if __name__ == "__main__":
    run_experiment()
