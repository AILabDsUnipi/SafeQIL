import torch as th


def dirichlet_kl_divergence_loss(alpha: th.Tensor, prior: th.Tensor) -> th.Tensor:
    """
    KL divergence between two dirichlet distributions
    The mean is alpha/(alpha+beta) and variance is alpha*beta/(alpha+beta)^2*(alpha+beta+1)
    There are multiple ways of modeling a dirichlet:
    1) by Laplace approximation with logistic normal: https://arxiv.org/pdf/1703.01488.pdf
    2) by directly modelling dirichlet parameters: https://arxiv.org/pdf/1901.02739.pdf
    code reference：
    1） https://github.com/sophieburkhardt/dirichlet-vae-topic-models
    2） https://github.com/is0383kk/Dirichlet-VAE
    """
    analytical_kld = th.lgamma(th.sum(alpha, dim=1)) - th.lgamma(th.sum(prior, dim=1))
    analytical_kld += th.sum(th.lgamma(prior), dim=1)
    analytical_kld -= th.sum(th.lgamma(alpha), dim=1)
    minus_term = alpha - prior

    digamma_term = th.digamma(alpha) - th.reshape(th.digamma(th.sum(alpha, dim=1)), shape=[alpha.shape[0], 1])
    test = th.sum(th.mul(minus_term, digamma_term), dim=1)
    analytical_kld += test

    return analytical_kld
