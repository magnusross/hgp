import logging
import os
import pickle
from distutils.dir_util import copy_tree
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

import hgp
from hgp.datasets.hamiltonians import load_system_from_name
from hgp.misc.plot_utils import (
    plot_comparison_traces,
    plot_learning_curve,
    plot_longitudinal,
)
from hgp.misc.torch_utils import numpy2torch, torch2numpy
from hgp.misc.train_utils import seed_everything
from hgp.misc.settings import settings

device = settings.device
if device.type == 'cuda':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

log = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def run_experiment(config: DictConfig):

    seed_everything(config.system.seed)
    # assert config.shooting == True

    times_ahead = 3

    system = load_system_from_name(config.system.system_name)(
        frequency_train=config.system.frequency_train,
        T_train=config.system.data_obs_T,
        frequency_test=config.system.frequency_test,
        T_test=times_ahead * config.system.data_obs_T,
        noise_var=config.system.data_obs_noise_var,
        noise_rel=config.system.noise_rel,
        seed=config.system.seed,
        N_x0s=config.system.num_traj,
        N_x0s_test=25,
    )

    system.scale_ts()
    system.scale_ys()

    if config.model.model_type == "hgp":
        model = hgp.models.builder.build_model(config, system.trn.ys)
    elif config.model.model_type == "hgp_subseq":
        model = hgp.models.builder.build_subsequence_model(config, system.trn.ys)
    elif config.model.model_type == "gpode":
        model = hgp.models.builder.build_gpode_model(config, system.trn.ys)
    elif config.model.model_type == "nn":
        model = hgp.models.builder.build_nn_model(config, system.trn.ys)
    else:
        raise ValueError("Model type not valid.")

    model, history = hgp.models.builder.init_and_fit(
        model, config, system.trn.ts, system.trn.ys, return_history=True
    )

    plot_learning_curve(
        history, save=os.path.join(os.getcwd(), f"lc_{config.model.name}.pdf")
    )
    model_vars = model.observation_likelihood.variance

    print("Generating predictions...")
    preds = hgp.models.builder.compute_test_predictions(
        model,
        system.x0_test,
        numpy2torch(system.tst.ts),
        eval_sample_size=config.eval_samples,
    )

    train_preds = (
        hgp.models.builder.compute_predictions(
            model,
            numpy2torch(system.trn.ts),
            eval_sample_size=config.eval_samples,
        )
        if config.model.model_type in ["hgp", "gpode"]
        else hgp.models.builder.compute_test_predictions(
            model,
            numpy2torch(system.trn.ys[:, 0, :]),
            numpy2torch(system.trn.ts),
            eval_sample_size=config.eval_samples,
        )
    )

    mll, mse, rel_err = hgp.models.builder.compute_summary(
        system.tst.ys,
        torch2numpy(preds),
        torch2numpy(model.observation_likelihood.variance),
        squeeze_time=False,
    )

    plot_longitudinal(
        system,
        torch2numpy(preds[:, : min(np.shape(preds)[1], 5)]),
        torch2numpy(model.observation_likelihood.variance),
        save=os.path.join(os.getcwd(), f"{config.model.name}_trajpost"),
    )

    plot_longitudinal(
        system,
        torch2numpy(train_preds[:, : max(np.shape(train_preds)[1], 5)]),
        torch2numpy(model.observation_likelihood.variance),
        save=os.path.join(os.getcwd(), f"train_{config.model.name}_trajpost"),
        test_true=(system.trn.ts, system.trn.ys),
    )

    res_dict = {}
    full_res_dict = {}

    full_res_dict["rmse"] = np.sqrt(mse)
    full_res_dict["mll"] = mll
    full_res_dict["rel_err"] = rel_err
    full_res_dict["preds"] = torch2numpy(preds)

    res_dict["rmse"] = np.sqrt(mse.mean())
    res_dict["mll"] = mll.mean()
    res_dict["rel_err"] = rel_err.mean()

    log.info(res_dict)

    full_res_dict["system"] = system
    with open(os.path.join(os.getcwd(), "metrics.pickle"), "wb") as handle:
        pickle.dump(full_res_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(os.getcwd(), "summary_metrics.pickle"), "wb") as handle:
        pickle.dump(res_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plot_comparison_traces(
        [preds],
        [model_vars],
        system,
        save=os.path.join(os.getcwd(), "comparison_traces.pdf"),
        names=[config.model.name],
    )

    if config.exp_dir:

        cwd = os.getcwd()
        last_path = cwd.split("/")[-1]
        main_path = f"data/results/{config.exp_dir}/"
        is_multi = last_path if len(last_path) <= 3 else ""

        res_path = hydra.utils.to_absolute_path(main_path + is_multi)
        filepath = Path(res_path)
        filepath.mkdir(parents=True, exist_ok=True)
        copy_tree(os.getcwd(), res_path)


if __name__ == "__main__":
    run_experiment()
