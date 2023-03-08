import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm


def log_lik(actual, predicted, noise_var):
    lik_samples = norm.logpdf(actual, loc=predicted, scale=noise_var**0.5)
    lik = logsumexp(lik_samples, 0, b=1 / float(predicted.shape[0]))
    return lik


def mse(actual, predicted):
    return np.power(actual - predicted.mean(0), 2)


def rel_err(actual, predicted):
    return np.sqrt(((actual - predicted.mean(0)) ** 2).mean(-1)) / (
        np.sqrt((predicted.mean(0) ** 2).mean(-1)) + np.sqrt((actual**2).mean(-1))
    )
