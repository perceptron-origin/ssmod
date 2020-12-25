"""
Test Utility Module
"""
import pytest
from numpy import array, allclose, isposinf, isneginf, ndarray
from ssmod.utils import (quadratic_fun,
                         get_default_gaussian,
                         get_default_uniform,
                         process_1darray,
                         process_gaussian,
                         process_uniform)


# pylint: disable=redefined-outer-name


@pytest.mark.parametrize("x", [array([1, 1, 1])])
def test_quadratic_fun(x):
    assert quadratic_fun(x) == 1.5


@pytest.mark.parametrize("size", [3])
def test_get_default_gaussian(size):
    g = get_default_gaussian(size)
    assert g.shape == (2, size)
    assert allclose(g[0], 0.0)
    assert all(isposinf(g[1]))


@pytest.mark.parametrize("size", [3])
def test_get_default_uniform(size):
    u = get_default_uniform(size)
    assert u.shape == (2, size)
    assert all(isneginf(u[0]))
    assert all(isposinf(u[1]))


@pytest.mark.parametrize("size", [3])
@pytest.mark.parametrize("x", [array([1, 2, 3, 4])])
def test_process_1darray_value_error(x, size):
    with pytest.raises(ValueError):
        process_1darray(x, size)


@pytest.mark.parametrize("size", [3])
@pytest.mark.parametrize("x", [[1, 2, 3]])
def test_process_1darray(x, size):
    x = process_1darray(x, size)
    assert isinstance(x, ndarray)


@pytest.mark.parametrize("mean", [0.0, [1, 2, 3, 4]])
@pytest.mark.parametrize("sd", [-1.0, [1, 2, 3, 4]])
@pytest.mark.parametrize("size", [3])
def test_process_gaussian_value_error(mean, sd, size):
    with pytest.raises(ValueError):
        process_gaussian(mean, sd, size)


@pytest.mark.parametrize("lb", [1.0, [1, 1, 1, 1]])
@pytest.mark.parametrize("ub", [0.0, [0, 0, 0, 0]])
@pytest.mark.parametrize("size", [3])
def test_process_uniform_value_error(lb, ub, size):
    with pytest.raises(ValueError):
        process_uniform(lb, ub, size)


@pytest.mark.parametrize("mean", [0.0, [0.0, 0.0, 0.0]])
@pytest.mark.parametrize("sd", [1.0, [1.0, 1.0, 1.0]])
@pytest.mark.parametrize("size", [3])
def test_process_gaussian(mean, sd, size):
    mean, sd = process_gaussian(mean, sd, size)
    assert mean.size == 3
    assert sd.size == 3
    assert allclose(mean, 0.0)
    assert allclose(sd, 1.0)


@pytest.mark.parametrize("lb", [0.0, [0.0, 0.0, 0.0]])
@pytest.mark.parametrize("ub", [1.0, [1.0, 1.0, 1.0]])
@pytest.mark.parametrize("size", [3])
def test_process_uniform(lb, ub, size):
    lb, ub = process_uniform(lb, ub, size)
    assert lb.size == 3
    assert ub.size == 3
    assert allclose(lb, 0.0)
    assert allclose(ub, 1.0)
