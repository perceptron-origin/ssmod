"""
Utility Functions
"""
from typing import Union, Tuple
from numpy import repeat, inf, ndarray, array, isscalar, asarray


def quadratic_fun(x: ndarray) -> float:
    return 0.5*sum(x**2)


def get_default_gaussian(size: int) -> ndarray:
    return array([[0.0]*size,
                  [inf]*size])


def get_default_uniform(size: int) -> ndarray:
    return array([[-inf]*size,
                  [inf]*size])


def process_1darray(x: Union[float, ndarray], size: int) -> ndarray:
    x = repeat(x, size) if isscalar(x) else asarray(x).ravel()
    if x.size != size:
        raise ValueError(f"array must be size {size}.")
    return x


def process_gaussian(mean: Union[float, ndarray],
                     sd: Union[float, ndarray],
                     size: int) -> Tuple[ndarray, ndarray]:
    mean = process_1darray(mean, size)
    sd = process_1darray(sd, size)
    if not all(sd > 0):
        raise ValueError("`sd` must be positive.")
    return mean, sd


def process_uniform(lb: Union[float, ndarray],
                    ub: Union[float, ndarray],
                    size: int) -> Tuple[ndarray, ndarray]:
    lb = process_1darray(lb, size)
    ub = process_1darray(ub, size)
    if not all(lb <= ub):
        raise ValueError("`lb` must be less or equal than `ub`.")
    return lb, ub
