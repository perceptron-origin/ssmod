"""
Test Data Class
"""
import pytest
from numpy import allclose
from ssmod.data import Data


# pylint: disable=redefined-outer-name


@pytest.fixture
def data():
    return Data(dim=3)


@pytest.mark.parametrize(("dim", "val"),
                         [(None, None),
                          (-1, None),
                          (3, [1, 2, 3, 4])])
def test_value_error(dim, val):
    with pytest.raises(ValueError):
        Data(dim=dim, val=val)


@pytest.mark.parametrize("val", [[1, 2, 3]])
def test_init(val):
    data = Data(val=val)
    assert data.dim == len(val)


@pytest.mark.parametrize("val", [[1, 2, 3]])
def test_val_setter(data, val):
    data.val = val
    assert allclose(data.val, val)
