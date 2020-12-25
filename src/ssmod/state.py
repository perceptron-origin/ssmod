"""
State Module
"""
from typing import Callable, Tuple, Union

from numpy import asarray, identity, ndarray, vstack

from ssmod.data import Data
from ssmod.linalg import CovarianceMatrix, ascovmat
from ssmod.utils import (get_default_gaussian, get_default_uniform,
                         process_gaussian, process_uniform, quadratic_fun)


class State(Data):
    """
    This class store the state information
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bounds = get_default_uniform(self.dim)
        self.prior = get_default_gaussian(self.dim)
        self.posterior = None

    def set_bounds(self,
                   lb: Union[float, ndarray],
                   ub: Union[float, ndarray]):
        self.bounds = vstack(process_uniform(lb, ub, self.dim))

    def set_prior(self,
                  mean: Union[float, ndarray],
                  sd: Union[float, ndarray]):
        self.prior = vstack(process_gaussian(mean, sd, self.dim))

    def set_posterior(self,
                      mean: Union[float, ndarray],
                      sd: Union[float, ndarray]):
        self.posterior = vstack(process_gaussian(mean, sd, self.dim))

    def __repr__(self) -> str:
        return f"State(dim={self.dim})"


class StateOperation:
    """
    This class define the operation on state.
    """

    def __init__(self,
                 opt_mat: ndarray,
                 cov_mat: CovarianceMatrix = None,
                 pen_fun: Callable = None):
        self.opt_mat = asarray(opt_mat)
        if self.opt_mat.ndim != 2:
            raise ValueError("`opt_mat` must be two dimensional array.")

        self.cov_mat = identity(self.shape[0]) if cov_mat is None else cov_mat
        self.cov_mat = ascovmat(self.cov_mat)
        if self.opt_mat.shape[0] != self.cov_mat.dim:
            raise ValueError("`opt_mat` and `cov_mat` dimension not match.")

        self.pen_fun = quadratic_fun if pen_fun is None else pen_fun
        if not callable(self.pen_fun):
            raise ValueError("`pen_fun` must be callable.")

    @property
    def shape(self) -> Tuple[int, int]:
        return self.opt_mat.shape

    def __call__(self, state: State) -> ndarray:
        return self.opt_mat.dot(state.val)

    def __repr__(self) -> str:
        return f"StateOperation(shape={self.shape})"
