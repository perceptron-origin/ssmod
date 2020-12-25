"""
Test State Class
"""
import pytest
from numpy import shares_memory, isposinf, isneginf, allclose
from numpy.random import seed, randn
from ssmod.state import State


# pylint: disable=redefined-outer-name


@pytest.fixture
def x():
    seed(123)
    return randn(3)


@pytest.fixture
def state(x):
    return State(val=x)


def test_state_ref(x):
    state = State(val=x)
    assert shares_memory(x, state.val)


def test_state_default_bounds(state):
    assert all(isneginf(state.bounds[0]))
    assert all(isposinf(state.bounds[1]))


def test_state_default_priors(state):
    assert all(isposinf(state.prior[1]))
    assert allclose(state.prior[0], 0.0)


@pytest.mark.parametrize("lb", [0.0, [0.0, 0.0, 0.0]])
@pytest.mark.parametrize("ub", [1.0, [1.0, 1.0, 1.0]])
def test_set_bounds(state, lb, ub):
    state.set_bounds(lb, ub)
    assert allclose(state.bounds[0], 0.0)
    assert allclose(state.bounds[1], 1.0)


@pytest.mark.parametrize("mean", [0.0, [0.0, 0.0, 0.0]])
@pytest.mark.parametrize("sd", [1.0, [1.0, 1.0, 1.0]])
def test_set_prior(state, mean, sd):
    state.set_prior(mean, sd)
    assert allclose(state.prior[0], 0.0)
    assert allclose(state.prior[1], 1.0)


@pytest.mark.parametrize("mean", [0.0, [0.0, 0.0, 0.0]])
@pytest.mark.parametrize("sd", [1.0, [1.0, 1.0, 1.0]])
def test_set_posterior(state, mean, sd):
    state.set_posterior(mean, sd)
    assert allclose(state.posterior[0], 0.0)
    assert allclose(state.posterior[1], 1.0)
