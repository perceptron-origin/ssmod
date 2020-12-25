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


def test_state_ref(x):
    state = State(val=x)
    assert shares_memory(x, state.val)


def test_state_dim(x):
    state = State(val=x)
    assert state.dim == x.size
