import torch


def build_J(D_in):
    assert D_in % 2 == 0
    I = torch.eye(D_in // 2)
    zeros = torch.zeros((D_in // 2, D_in // 2))
    zI = torch.hstack((zeros, I))
    mIz = torch.hstack((-I, zeros))
    return torch.vstack((zI, mIz))
