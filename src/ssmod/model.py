"""
Model Module
"""
from typing import List
from numpy import ndarray, reshape, hstack
from ssmod.data import Data
from ssmod.state import State, StateOperation


class StateSpaceModel:
    """
    This class construct the statistical model and optimization interface.
    """

    def __init__(self,
                 process: List[StateOperation],
                 measure: List[StateOperation],
                 measurements: List[Data],
                 states: List[State] = None):
        self.process = process
        self.measure = measure
        self.measurements = measurements

        self.dim_state = self.process[0].shape[0]
        self.dim_measurement = self.measure[0].shape[0]
        self.size = len(self.process)

        if states is None:
            states = [
                State(dim=self.dim_state)
                for i in range(self.size)
            ]
        self.states = states
        self.prior = hstack([state.prior for state in states])
        self.bounds = hstack([state.bounds for state in states])

    def array_to_states(self, x: ndarray):
        x = reshape(x, (self.size, self.dim_state))
        for i, state in enumerate(self.states):
            state.val = x[i]

    def objective(self, x: ndarray) -> float:
        self.array_to_states(x)
        v = 0.0
        # process
        for i in range(self.size - 1):
            v += self.process[i].pen_fun(
                self.process[i].cov_mat.inv_sqrt_mat.dot(
                    self.states[i + 1] - self.process[i](self.states[i])
                )
            )

        # measure
        for i in range(self.size):
            v += self.measure[i].pen_fun(
                self.measure[i].cov_mat.inv_sqrt_mat.dot(
                    self.measurements[i] - self.measure[i](self.states[i])
                )
            )

        # prior
        v += 0.5*sum((x - self.prior[0])**2/self.prior[1]**2)

        return v
