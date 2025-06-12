import math

import numpy as np
import torch
from properscoring import crps_gaussian
from scipy.stats import norm


def gaussian_sample(mu, sigma):
    '''
    Gaussian Sample
    Args:
    ytrue (array like)
    mu (array like)
    sigma (array like): standard deviation

    gaussian maximum likelihood using log
        l_{G} (z|mu, sigma) = (2 * pi * sigma^2)^(-0.5) * exp(- (z - mu)^2 / (2 * sigma^2))
    '''
    # likelihood = (2 * np.pi * sigma ** 2) ** (-0.5) * \
    #         torch.exp((- (ytrue - mu) ** 2) / (2 * sigma ** 2))
    # return likelihood
    gaussian = torch.distributions.normal.Normal(mu, sigma)
    ypred = gaussian.rsample()
    return ypred


def negative_binomial_sample(mu, alpha):
    '''
    Negative Binomial Sample
    Args:
    ytrue (array like)
    mu (array like)
    alpha (array like)

    maximuze log l_{nb} = log Gamma(z + 1/alpha) - log Gamma(z + 1) - log Gamma(1 / alpha)
                - 1 / alpha * log (1 + alpha * mu) + z * log (alpha * mu / (1 + alpha * mu))

    minimize loss = - log l_{nb}

    Note: torch.lgamma: log Gamma function
    '''
    var = mu + mu * mu * alpha
    ypred = mu + torch.randn(mu.size()).to(mu.device) * torch.sqrt(var)
    return ypred


def gaussian_likelihood_loss(target, mu, sigma,eps=1e-6):
    '''
    Gaussian Liklihood Loss
    Args:
    target (tensor): true observations, shape (num_ts, num_periods)
    mu (tensor): mean, shape (num_ts, num_periods)
    sigma (tensor): standard deviation, shape (num_ts, num_periods)

    likelihood:
    (2 pi sigma^2)^(-1/2) exp(-(target - mu)^2 / (2 sigma^2))

    log likelihood:
    -1/2 * (log (2 pi) + 2 * log (sigma)) - 1/2 *  (target - mu)^2 / (sigma^2) + constant  # constant对结果没有影响
    '''
    sigma = sigma.clone()
    with torch.no_grad():
        sigma.clamp_(min=eps)

    # negative_likelihood = torch.log(sigma + 1) + (target - mu) ** 2 / (2 * sigma ** 2) + 6  # deepar
    # negative_likelihood = 0.5 * (torch.log(sigma+1) + (mu - target)**2 / sigma) + 0.5 * math.log(2 * torch.pi)  # sigma1
    # negative_likelihood = 0.5 * (torch.log(sigma) + (mu - target)**2 / sigma) + 0.5 * math.log(2 * torch.pi)  # g  torch.nn.GaussianNLLLoss
    negative_likelihood = torch.log(sigma) + 0.5 * (mu - target)**2 / sigma ** 2 + math.log(2 * torch.pi)  # g  torch.nn.GaussianNLLLoss

    return negative_likelihood.mean()


def negative_binomial_loss(ytrue, mu, alpha):
    '''
    Negative Binomial Sample
    Args:
    ytrue (array like)
    mu (array like)
    alpha (array like)

    maximuze log l_{nb} = log Gamma(z + 1/alpha) - log Gamma(z + 1) - log Gamma(1 / alpha)
                - 1 / alpha * log (1 + alpha * mu) + z * log (alpha * mu / (1 + alpha * mu))

    minimize loss = - log l_{nb}

    Note: torch.lgamma: log Gamma function
    '''
    batch_size, seq_len,_ = ytrue.size()
    likelihood = torch.lgamma(ytrue + 1. / alpha) - torch.lgamma(ytrue + 1) - torch.lgamma(1. / alpha) \
        - 1. / alpha * torch.log(1 + alpha * mu) \
        + ytrue * torch.log(alpha * mu / (1 + alpha * mu))
    return - likelihood.mean()


def MAPE(ytrue, ypred):
    ytrue = np.array(ytrue).ravel() + 1e-4
    ypred = np.array(ypred).ravel()
    return np.mean(np.abs((ytrue - ypred) / ytrue))

def gaussian_nll(y_true, mu, sigma):
    return np.mean(0.5 * np.log(2 * np.pi * sigma**2) + ((y_true - mu)**2) / (2 * sigma**2))

def crps_score(y_true, mu, sigma):
    return np.mean(crps_gaussian(y_true, mu, sigma))


def picp(y_true, mu, sigma, alpha=0.9):
    z = norm.ppf(1 - (1 - alpha) / 2)
    lower = mu - z * sigma
    upper = mu + z * sigma
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    return coverage

def piw(mu, sigma, alpha=0.9):
    z = norm.ppf(1 - (1 - alpha) / 2)
    width = 2 * z * sigma
    return np.mean(width)