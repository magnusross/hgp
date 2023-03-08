from hgp.misc.settings import settings
import numpy as np
import torch

device = settings.device
dtype = settings.torch_float


def numpy2torch(x):
    return (
        torch.tensor(x, dtype=dtype).to(device)
        if type(x) is np.ndarray
        else x.to(device)
    )


def torch2numpy(x):
    return x if type(x) is np.ndarray else x.detach().cpu().numpy()


def restore_model(model, filename):
    checkpt = torch.load(filename, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpt["state_dict"])
    return model


def save_model(model, filename):
    torch.save({"state_dict": model.state_dict()}, filename)


def save_model_optimizer(model, optimizer, filename):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        filename,
    )


def insert_zero_t0(ts):
    return torch.cat([torch.tensor([0.0]), ts + ts[1] - ts[0]])


def compute_ts_dense(ts, ts_dense_scale):
    """
    Given a time sequence ts, this makes it dense by adding intermediate time points.
    @param ts: time sequence
    @param ts_dense_scale: dense factor
    @return: dense time sequence
    """
    if ts_dense_scale > 1:
        ts_dense = torch.cat(
            [
                torch.linspace(t1, t2, ts_dense_scale)[:-1]
                for (t1, t2) in zip(ts[:-1], ts[1:])
            ]
            + [ts[-1:]]
        )
    else:
        ts_dense = ts
    return ts_dense
