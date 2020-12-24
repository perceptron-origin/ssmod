"""
Test State Class
"""
import pytest
from numpy import shares_memory
from numpy.random import seed, randn
from ssmod.state import State


# pylint: disable=redefined-outer-name


@pytest.fixture
def x():
    seed(123)
    return randn(3)


@pytest.fixture
def y():
    seed(456)
    return randn(2)


def test_state_ref(x, y):
    state = State(x, y)
    assert shares_memory(x, state.x)
    assert shares_memory(y, state.y)


def test_state_dim(x, y):
    state = State(x, y)
    assert state.dim_x == x.size
    assert state.dim_y == y.size
