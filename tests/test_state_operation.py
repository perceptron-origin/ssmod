"""
Test State Operation Class
"""
import pytest
from numpy import ones, identity, allclose
from numpy.random import seed, randn
from ssmod.state import State, StateOperation


# pylint: disable=redefined-outer-name


@pytest.fixture
def state():
    seed(123)
    return State(val=randn(4))


@pytest.mark.parametrize("opt_mat", [ones(4)])
def test_value_error_opt_mat(opt_mat):
    with pytest.raises(ValueError):
        StateOperation(opt_mat)


@pytest.mark.parametrize("opt_mat", [ones((1, 4))])
@pytest.mark.parametrize("cov_mat", [identity(2)])
def test_value_error_cov_mat(opt_mat, cov_mat):
    with pytest.raises(ValueError):
        StateOperation(opt_mat, cov_mat)


@pytest.mark.parametrize("opt_mat", [ones((1, 4))])
@pytest.mark.parametrize("pen_fun", [1])
def test_value_error_pen_fun(opt_mat, pen_fun):
    with pytest.raises(ValueError):
        StateOperation(opt_mat, pen_fun=pen_fun)


@pytest.mark.parametrize("opt_mat", [ones((1, 4))])
def test_default_values(opt_mat):
    opt = StateOperation(opt_mat)
    assert allclose(opt.cov_mat.mat, identity(1))
    assert allclose(opt.pen_fun(ones(1)), 0.5)


@pytest.mark.parametrize("opt_mat", [ones((1, 4))])
def test_call(opt_mat, state):
    opt = StateOperation(opt_mat)
    assert allclose(opt(state), sum(state.val))
