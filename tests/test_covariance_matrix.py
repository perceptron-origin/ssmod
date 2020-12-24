"""
Test Covariance Matrix Class
"""
import pytest
from numpy import allclose, eye
from numpy.random import seed, randn
from ssmod.linalg import CovarianceMatrix

# pylint: disable=redefined-outer-name


@pytest.fixture
def mat():
    seed(123)
    sqrt_mat = randn(3, 3)
    return sqrt_mat.dot(sqrt_mat.T)


@pytest.mark.parametrize("mat", [randn(3, 3, 3),
                                 randn(2, 3)])
def test_value_error(mat):
    with pytest.raises(ValueError):
        CovarianceMatrix(mat)


def test_shape(mat):
    cov_mat = CovarianceMatrix(mat)
    assert cov_mat.shape == mat.shape


def test_dim(mat):
    cov_mat = CovarianceMatrix(mat)
    assert cov_mat.dim == mat.shape[0]


def test_sqrt_mat(mat):
    cov_mat = CovarianceMatrix(mat)
    assert allclose(cov_mat.sqrt_mat.dot(cov_mat.sqrt_mat.T), mat)


def test_inv_sqrt_mat(mat):
    cov_mat = CovarianceMatrix(mat)
    assert allclose(cov_mat.sqrt_mat.dot(cov_mat.inv_sqrt_mat),
                    eye(cov_mat.dim))
