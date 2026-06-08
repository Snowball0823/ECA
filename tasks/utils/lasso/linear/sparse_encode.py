import warnings
import torch

from .utils import lstsq, ridge
from .solvers import (coord_descent, gpsr_basic, iterative_ridge, ista,
                      interior_point, split_bregman, orthant_wise_newton)

_init_defaults = {
    'ista': 'zero',
    'cd': 'zero',
    'gpsr': 'zero',
    'iter-ridge': 'ridge',
    'interior-point': 'ridge',
    'split-bregman': 'zero',
    'own': 'zero'
}


def initialize_code(x, weight, alpha, mode):
    n_samples = x.size(0)
    n_components = weight.size(1)
    if mode == 'zero':
        z0 = x.new_zeros(n_samples, n_components)
    elif mode == 'unif':
        z0 = x.new(n_samples, n_components).uniform_(-0.1, 0.1)
    elif mode == 'lstsq':
        z0 = lstsq(x.T, weight).T
    elif mode == 'ridge':
        z0 = ridge(x.T, weight, alpha=alpha).T
    elif mode == 'transpose':
        z0 = torch.matmul(x, weight)
    else:
        raise ValueError("invalid init parameter '{}'.".format(mode))
    return z0


def sparse_encode(x, 
                  weight, 
                  *,
                  alpha=1.0, 
                  z0=None, 
                  algorithm='ista', 
                  init=None,
                  maxiter=100,
                  tol=1e-5,
                  verbose=False,
                  positive_code=False,
                  **kwargs):
    '''
    args:
        x: input torch.Tensor data => [n_samples, n_features]
        weight: dictionary => [n_features, n_components]
        z0: for the initial sparse code, set to `None` in minibatch dictionary learning. 
            Only be used when fit the whole dict by the whole dataset.
        init: the method of init methods ['zero', 'unif', 'lstsq', 'ridge', 'transpose'].
        maxiter: the max iteration of fitting.
        tol: for the early stopping
        verbose: if show the process of training.
        positive_code: enforce the sparse code to be positive >0
    kwargs:
        # algorithm => 'gpsr'
        stop_criterion:
        miniter:
        init:
        continuation:
        debias:
        -> More to be done
        # algorithm => 'iter-ridge'
        tikhonov:
        eps:
        line_search:
        cg:
        cg_options:
        # algorithm => 'ista'
        fast: default -> True
        lr: default -> 'auto'
        backtrack: default -> False
        eta_backtrack: default -> 1.5
    return:
        z: the sparse code => [n_samples, n_components]
    '''
    n_samples = x.size(0)
    n_components = weight.size(1)

    # initialize code variable
    if z0 is not None:
        assert z0.shape == (n_samples, n_components)
    else:
        if init is None:
            init = _init_defaults.get(algorithm, 'zero')
        elif init == 'zero' and algorithm == 'iter-ridge':
            warnings.warn("Iterative Ridge should not be zero-initialized.")
        z0 = initialize_code(x, weight, alpha, mode=init)

    kwargs.update({'maxiter':maxiter, 'tol':tol, 'verbose':verbose, "positive_code":positive_code})
    # perform inference
    # => common kwargs: x, weight, alpha, maxiter, verbose
    if algorithm == 'cd':
        # x, W, z0=None, alpha=1.0, maxiter=1000, tol=1e-6, verbose=False
        # kwargs: positive_code=False
        z = coord_descent(x, weight, z0, alpha, **kwargs)
    elif algorithm == 'gpsr':
        A = lambda v: torch.mm(v, weight.T)
        AT = lambda v: torch.mm(v, weight)
        # y, A, tau, AT=None, x0=None, stop_criterion=3, tol=1e-2,
        #    maxiter=1000, miniter=5, init=0, continuation=False,
        #    debias=False, verbose=0, **kwargs 
        # kwargs: positive_code=False
        z = gpsr_basic(x, A, tau=alpha, AT=AT, x0=z0, **kwargs)
    elif algorithm == 'iter-ridge':
        # z0, x, weight, alpha=1.0, tol=1e-5, tikhonov=1e-4, eps=None,
        #             maxiter=10, line_search=True, cg=False, cg_options=None,
        #             verbose=False
        z = iterative_ridge(z0, x, weight, alpha, **kwargs)
    elif algorithm == 'ista':
        # x, z0, weight, alpha=1.0, fast=True, lr='auto', maxiter=10,
        #  tol=1e-5, backtrack=False, eta_backtrack=1.5, verbose=False
        # kwargs: positive_code=False
        z = ista(x, z0, weight, alpha, **kwargs)
    elif algorithm == 'interior-point':
        # x, weight, z0=None, alpha=1.0, maxiter=20, barrier_init=0.1,
        #            tol=1e-2, eps=1e-5, verbose=False
        z, _ = interior_point(x, weight, z0, alpha, **kwargs)
    elif algorithm == 'split-bregman':
        # A, y, x0=None, alpha=1.0, lambd=1.0, maxiter=20, niter_inner=5,
        #           tol=1e-10, tau=1., verbose=False
        # kwargs: positive_code=False
        z, _ = split_bregman(weight, x, z0, alpha, **kwargs)
    elif algorithm == 'own':
        # weight, x, z0, alpha=1., lr=1., maxiter=20, xtol=1e-5,
        # line_search='brent', ls_options=None, verbose=0
        # kwargs: positive_code=False
        z = orthant_wise_newton(weight, x, z0, alpha, **kwargs)
    else:
        raise ValueError("invalid algorithm parameter '{}'.".format(algorithm))

    return z