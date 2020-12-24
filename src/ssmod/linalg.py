"""
Linear Algebra Module
"""
from numpy import ndarray
from numpy.linalg import cholesky, inv


class CovarianceMatrix:
    """
    This class will take in a square matrix

    * compute the Cholesky factorization
    * compute the inverse of Cholesky factorization
    """

    def __init__(self, mat: ndarray):
        if mat.ndim != 2:
            raise ValueError("`mat` must be a two dimensional array.")
        if mat.shape[0] != mat.shape[1]:
            raise ValueError("`mat` must be a square matrix.")

        self.mat = mat
        self.sqrt_mat = cholesky(self.mat)
        self.inv_sqrt_mat = inv(self.sqrt_mat)

    @property
    def shape(self) -> int:
        return self.mat.shape

    @property
    def dim(self) -> int:
        return self.mat.shape[0]

    def __repr__(self) -> str:
        return f"CovarianceMatrix(dim={self.dim})"
