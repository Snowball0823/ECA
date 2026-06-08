"""
 Copyright (c) 2026, Jiangtao Kong.
 Contact: Jiangtao Kong <tinysnowball0823@gmail.com>
 Released for non-commercial research use only.
 For license details, see the LICENSE and NOTICE files in the repo root.
"""
import copy
from copy import deepcopy
import string
import torch.distributed.nn
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg

from .lasso.linear import sparse_encode

LASSO_SOLVER = ['omp', 'cd', 'gpsr', 'iter-ridge',
                'ista', 'fista', 'interior-point', 'split-bregman', 'own']


class MiniBatchScaleCodeDictionaryLearner(object):
    """
    TODO: Add more fit methods later; Add parallel into the fit algorithm, use n_job;

    Inspired by sklearn->dictionary_learning

    Mini-batch dictionary learning.

    Finds a dictionary (a set of atoms) that performs well at sparsely
    encoding the fitted data.

    Solves the optimization problem::

       (U^*,V^*) = argmin 0.5 || X - U V ||_Fro^2 + alpha * || U ||_1,1
                    (U,V)
                    with || V_k ||_2 <= 1 for all  0 <= k < n_components

    ||.||_Fro stands for the Frobenius norm and ||.||_1,1 stands for
    the entry-wise matrix norm which is the sum of the absolute values
    of all the entries in the matrix.

    Read more in the :ref:`User Guide <DictionaryLearning>`.

    Parameters
    ----------
    n_components : int, default=None
        Number of dictionary elements to extract.

    alpha : float, default=1
        Sparsity controlling parameter.

    n_iter : int, default=1000
        Total number of iterations over data batches to perform.

        .. deprecated:: 1.1
           ``n_iter`` is deprecated in 1.1 and will be removed in 1.4. Use
           ``max_iter`` instead.

    fit_algorithm : {'lars', 'cd'}, default='lars'
        The algorithm used:

        - `'lars'`: uses the least angle regression method to solve the lasso
          problem (`linear_model.lars_path`)
        - `'cd'`: uses the coordinate descent method to compute the
          Lasso solution (`linear_model.Lasso`). Lars will be faster if
          the estimated components are sparse.

    batch_size : int, default=256
        Number of samples in each mini-batch.

        .. versionchanged:: 1.3
           The default value of `batch_size` changed from 3 to 256 in version 1.3.

    shuffle : bool, default=True
        Whether to shuffle the samples before forming batches.

    dict_init : ndarray of shape (n_components, n_features), default=None
        Initial value of the dictionary for warm restart scenarios.

    verbose : bool or int, default=False
        To control the verbosity of the procedure.

    split_sign : bool, default=False
        Whether to split the sparse feature vector into the concatenation of
        its negative part and its positive part. This can improve the
        performance of downstream classifiers.

    positive_dict : bool, default=False
        Whether to enforce positivity when finding the dictionary.

        .. versionadded:: 0.20

    transform_max_iter : int, default=1000
        Maximum number of iterations to perform if `algorithm='lasso_cd'` or
        `'lasso_lars'`.

        .. versionadded:: 0.22

    tol : float, default=1e-3
        Control early stopping based on the norm of the differences in the
        dictionary between 2 steps. Used only if `max_iter` is not None.

        To disable early stopping based on changes in the dictionary, set
        `tol` to 0.0.

        .. versionadded:: 1.1

    max_no_improvement : int, default=10
        Control early stopping based on the consecutive number of mini batches
        that does not yield an improvement on the smoothed cost function. Used only if
        `max_iter` is not None.

        To disable convergence detection based on cost function, set
        `max_no_improvement` to None.

        .. versionadded:: 1.1

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Components extracted from the data.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : int
        Number of iterations over the full dataset.

    n_steps_ : int
        Number of mini-batches processed.

        .. versionadded:: 1.1

    See Also
    --------
    DictionaryLearning : Find a dictionary that sparsely encodes data.
    MiniBatchSparsePCA : Mini-batch Sparse Principal Components Analysis.
    SparseCoder : Find a sparse representation of data from a fixed,
        precomputed dictionary.
    SparsePCA : Sparse Principal Components Analysis.

    References
    ----------

    J. Mairal, F. Bach, J. Ponce, G. Sapiro, 2009: Online dictionary learning
    for sparse coding (https://www.di.ens.fr/sierra/pdfs/icml09.pdf)

    Examples
    --------
    TODO
    """

    def __init__(
        self,
        n_components=None,
        *,
        alpha=1.0,
        n_iter=1000,
        fit_algorithm="ista",
        batch_size=256,
        shuffle=True,
        constrained=True,
        dict_init=None,
        verbose=False,
        split_sign=False,
        positive_dict=False,
        tol=1e-3,
        max_no_improvement=10,
        device='cpu',
        num_process=-1,
        **kwargs
    ):
        if isinstance(device, str):
            assert device in ['cpu','cuda','gpu'], "Invalid device {}, device shoul be one of ['cpu','cuda','gpu'].".format(device)
            if device == 'gpu':
                device = 'cuda'
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            print("Invalid device type, set to `cpu` default.")
            self.device = torch.device('cpu')
        # only be used when self.device is 'cpu'
        # => TODO: to finish it
        self.num_process = num_process
        # for transform to sparse code
        self.alpha = alpha
        # => set for solve the LASSO problem: cd, lars, isat, fisat, ...
        self.fit_algorithm = fit_algorithm
        assert fit_algorithm in LASSO_SOLVER, "The fit_algorithm: {} is not available now.".format(fit_algorithm)
        # => while it is used by dictionary learning, set to `False` enforced
        self.split_sign = split_sign
        # for dictionary
        self.constrained = constrained
        self.n_components = n_components
        self.dict_init = dict_init
        # for fit the dictionary (not for minibatch_fit)
        self.n_iter = n_iter
        self.batch_size = batch_size
        # => will be used later
        self.verbose = verbose
        self.shuffle = shuffle
        self.split_sign = split_sign
        # make sure all dict element are positive
        self.positive_dict = positive_dict
        # for early stop
        self.max_no_improvement = max_no_improvement
        self.tol = tol
        # set for internal artributs
        self.dictionary = None
        self.inner_step = 0


    def _check_params(self, X:torch.Tensor):
        # n_components
        self._n_components = self.n_components
        if self._n_components is None:
            self._n_components = X.shape[1]

        # batch_size
        self._batch_size = min(self.batch_size, X.shape[0])


    def _initialize_dict(self, X: torch.Tensor):
        """Initialization of the dictionary."""
        # _check_params first
        _, n_features = X.shape
        
        if self.dict_init is not None:
            dictionary = self.dict_init
        else:
            # init dictionary with orthogonal
            dictionary = torch.empty(n_features, self._n_components)
            nn.init.orthogonal_(dictionary)

        if self.constrained:
            dictionary = F.normalize(dictionary, dim=0)

        dictionary = dictionary.to(self.device)
        setattr(self, 'dictionary', dictionary)
        return dictionary
    

    def _init_inter_state(self, train_data:torch.Tensor):
        # _check_params first
        _, n_feature = train_data.shape
        self._A = torch.zeros((self._n_components, self._n_components), dtype=train_data.dtype).to(self.device)
        self._B = torch.zeros((n_feature, self._n_components), dtype=train_data.dtype).to(self.device)


    def _update_inner_stats(self, X: torch.Tensor, code: torch.Tensor):
        """Update the inner stats inplace."""
        batch_size = X.shape[0]
        if self.inner_step < batch_size - 1:
            theta = (self.inner_step + 1) * batch_size
        else:
            theta = batch_size**2 + self.inner_step + 1 - batch_size
        beta = (theta + 1 - batch_size) / (theta + 1)

        self._A *= beta
        self._A += code.T @ code / batch_size
        self._B *= beta
        self._B += X.T @ code / batch_size


    # TODO: multprocess
    def _update_dict(self,
                     X: torch.Tensor,
                     code: torch.Tensor,
                     A=None,
                     B=None,):
        
        n_samples, n_components = code.shape
        dictionary = getattr(self, 'dictionary')

        if A is None:
            A = code.T @ code
        if B is None:
            B = X.T @ code

        n_unused = 0
        # TODO: parallel do the check
        for k in range(n_components):
            if A[k, k] > 1e-6:
                # 1e-6 is arbitrary but consistent with the spams implementation
                dictionary[:, k] += (B[:, k] - A[k] @ dictionary.T) / A[k, k]
            else:
                # k_th atom is almost never used -> sample a new one from the data
                newd = X[torch.randint(0, n_samples, (1,)).item()]
                # add small noise to avoid making the sparse coding ill conditioned
                noise_level = 0.01 * (newd.std() or 1)  # avoid std == 0
                noise = torch.randn_like(newd) * noise_level
                dictionary[:, k] = newd + noise
                code[:, k].zero_()
                n_unused += 1

            if self.positive_dict:
                dictionary[:, k].clamp_(min=0)
            # project to ||V_k|| <= 1
            dictionary[:, k] /= max(linalg.norm(dictionary[:, k]), 1)
        if self.verbose and n_unused > 0:
            print(f"{n_unused} unused atoms resampled.")

    
    def lasso_loss(self, X, code):
        dictionary = getattr(self, 'dictionary')
        X_hat = torch.matmul(code, dictionary.T)
        loss = 0.5 * (X - X_hat).pow(2).sum() + self.alpha * code.abs().sum()
        return loss / X.size(0)

    def _minibatch_step(self, X, **solver_kwargs):
        """Perform the update on the dictionary for one minibatch."""
        dictionary = getattr(self, 'dictionary')

        # Compute code for this batch
        # infer sparse coefficients and compute loss
        # TODO: Lars using multiprocess in CPU or modify in torch later
        code = sparse_encode(
            X, 
            dictionary, 
            alpha=self.alpha, 
            algorithm=self.fit_algorithm, 
            **solver_kwargs
        )

        loss = self.lasso_loss(X, code)

        # Update inner stats
        self._update_inner_stats(X, code)
        # Update dictionary
        self._update_dict(X, code, self._A, self._B)

        return loss

    def _check_convergence(
        self, X, batch_cost, new_dict, old_dict, n_samples, step, n_steps
    ):
        """Helper function to encapsulate the early stopping logic.

        Early stopping is based on two factors:
        - A small change of the dictionary between two minibatch updates. This is
          controlled by the tol parameter.
        - No more improvement on a smoothed estimate of the objective function for a
          a certain number of consecutive minibatch updates. This is controlled by
          the max_no_improvement parameter.
        """
        batch_size = X.shape[0]

        # counts steps starting from 1 for user friendly verbose mode.
        step = step + 1

        # Ignore 100 first steps or 1 epoch to avoid initializing the ewa_cost with a
        # too bad value
        if step <= min(100, n_samples / batch_size):
            if self.verbose:
                print(f"Minibatch step {step}/{n_steps}: mean batch cost: {batch_cost}")
            return False

        # Compute an Exponentially Weighted Average of the cost function to
        # monitor the convergence while discarding minibatch-local stochastic
        # variability: https://en.wikipedia.org/wiki/Moving_average
        if self._ewa_cost is None:
            self._ewa_cost = batch_cost
        else:
            alpha = batch_size / (n_samples + 1)
            alpha = min(alpha, 1)
            self._ewa_cost = self._ewa_cost * (1 - alpha) + batch_cost * alpha

        if self.verbose:
            print(
                f"Minibatch step {step}/{n_steps}: mean batch cost: "
                f"{batch_cost}, ewa cost: {self._ewa_cost}"
            )

        # Early stopping based on change of dictionary
        dict_diff = linalg.norm(new_dict - old_dict) / self._n_components
        if self.tol > 0 and dict_diff <= self.tol:
            if self.verbose:
                print(f"Converged (small dictionary change) at step {step}/{n_steps}")
            return True

        # Early stopping heuristic due to lack of improvement on smoothed
        # cost function
        if self._ewa_cost_min is None or self._ewa_cost < self._ewa_cost_min:
            self._no_improvement = 0
            self._ewa_cost_min = self._ewa_cost
        else:
            self._no_improvement += 1

        if (
            self.max_no_improvement is not None
            and self._no_improvement >= self.max_no_improvement
        ):
            if self.verbose:
                print(
                    "Converged (lack of improvement in objective function) "
                    f"at step {step}/{n_steps}"
                )
            return True

        return False


    def minibatch_fit(self, X, **solver_kwargs):
        """Update the model using the data in X as a mini-batch.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Return the instance itself.
        """

        dictionary = getattr(self, "dictionary")
        if dictionary is None:
            self.init_dictionary(X)
            self.inner_step = 0

        loss = self._minibatch_step(X, **solver_kwargs)

        self.inner_step += 1

        return loss


    def init_dictionary(self, X: torch.Tensor):
        # This instance has not been fitted yet (fit or partial_fit)
        self._check_params(X)
        self._initialize_dict(X)
        self._init_inter_state(X)



    def transfer(self, X, **solver_kwargs):
        dictionary = getattr(self, "dictionary")
        assert dictionary is not None, "Using `minibatch_fit` first."
        code = sparse_encode(
            X, 
            dictionary, 
            alpha=self.alpha, 
            algorithm=self.fit_algorithm, 
            **solver_kwargs
        )
        return code


    def clear_inner_state(self):
        _A = torch.zeros_like(self._A)
        _B = torch.zeros_like(self._B)
        del(self._A, self._B)
        self._A = _A
        self._B = _B
        self.inner_step = 0


    def get_inner_state(self):
        return {'inner_step':self.inner_step, '_A':self._A.detach().cpu(), '_B':self._B.detach().cpu()}


    def reload_inner_state(self, state_dict:dict):
        assert self.dict_init is not None, 'Using reload_inner_state, you must give the previous dictonary as initial dictionary.'
        assert '_A' in state_dict and '_B' in state_dict, "Missing key of 'inner_state', using 'get_inner_state' to save."
        for key in state_dict:
            setattr(self, key, state_dict[key])

        self._A = self._A.to(self.device)
        self._B = self._B.to(self.device)

        dictionary = self.dict_init
        if self.constrained:
            dictionary = F.normalize(dictionary, dim=0)

        dictionary = dictionary.to(self.device)
        setattr(self, 'dictionary', dictionary)
        

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        assert hasattr(self, "dictionary")
        dictionary = getattr(self, "dictionary")
        return dictionary.shape[1]

