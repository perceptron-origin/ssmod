"""
Linear Algebra Module
"""
from typing import Tuple, Union
from numpy import ndarray, asarray
from numpy.linalg import cholesky, inv


class CovarianceMatrix:
    """
    This class will take in a square matrix

    * compute the Cholesky factorization
    * compute the inverse of Cholesky factorization
    """

    def __init__(self, mat: ndarray):
        mat = asarray(mat)
        if mat.ndim != 2:
            raise ValueError("`mat` must be a two dimensional array.")
        if mat.shape[0] != mat.shape[1]:
            raise ValueError("`mat` must be a square matrix.")

        self._mat = mat
        self._sqrt_mat = cholesky(self._mat)
        self._inv_sqrt_mat = inv(self._sqrt_mat)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.mat.shape

    @property
    def dim(self) -> int:
        return self.mat.shape[0]

    @property
    def mat(self) -> ndarray:
        return self._mat

    @mat.setter
    def mat(self, mat: ndarray):
        self.__init__(mat)

    @property
    def sqrt_mat(self) -> ndarray:
        return self._sqrt_mat

    @property
    def inv_sqrt_mat(self) -> ndarray:
        return self._inv_sqrt_mat

    def __repr__(self) -> str:
        return f"CovarianceMatrix(dim={self.dim})"


def ascovmat(mat: Union[ndarray, CovarianceMatrix]) -> CovarianceMatrix:
    if not isinstance(mat, CovarianceMatrix):
        mat = CovarianceMatrix(mat)
    return mat
