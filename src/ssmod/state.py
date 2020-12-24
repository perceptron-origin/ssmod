"""
State Module
"""
from typing import Callable, Tuple
from numpy import ndarray, asarray, identity
from ssmod.linalg import CovarianceMatrix, ascovmat


class State:
    """
    This class store the state information
    """

    def __init__(self, x: ndarray, y: ndarray):
        self.x = asarray(x).ravel()
        self.y = asarray(y).ravel()

    @property
    def dim_x(self) -> int:
        return self.x.size

    @property
    def dim_y(self) -> int:
        return self.y.size

    def __repr__(self) -> str:
        return f"State(dim_x={self.dim_x}, dim_y={self.dim_y})"


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
        return self.opt_mat.dot(state.x)

    def __repr__(self) -> str:
        return f"StateOperation(shape={self.shape})"


def quadratic_fun(x: ndarray) -> float:
    return 0.5*sum(x**2)
