"""
Data Module
"""
from typing import Union
from numpy import ndarray, asarray


class Data:
    """
    This class is a data container
    """

    def __init__(self, dim: int = None, val: ndarray = None):
        if dim is None and val is None:
            raise ValueError("Must provide `dim` or `val`.")
        if dim is None:
            dim = len(val)
        if not (isinstance(dim, int) and dim > 0):
            raise ValueError("`dim` has to be a positive integer.")

        self._val = None
        self.dim = dim
        self.val = val

    @property
    def val(self) -> ndarray:
        return self._val

    @val.setter
    def val(self, val: Union[None, ndarray]):
        if val is not None:
            val = asarray(val).ravel()
            if val.size != self.dim:
                raise ValueError(f"`val` must be size {self.dim}.")
        self._val = val

    def __repr__(self) -> str:
        return f"Data(dim={self.dim})"
