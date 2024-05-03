from abc import ABC
from enum import Enum
import numpy as np
import pandas as pd

from psyke.extraction.hypercubic import Grid
from psyke.utils import Target


class Objective(Enum):
    MODEL = 1,
    DATA = 2


class Optimizer:
    def __init__(self, dataframe: pd.DataFrame, output: Target = Target.CONSTANT, max_error_increase: float = 1.2,
                 min_rule_decrease: float = 0.9, readability_tradeoff: float = 0.1, patience: int = 5,
                 normalization=None, discretization=None):
        self.dataframe = dataframe
        self.output = output
        self.max_error_increase = max_error_increase
        self.min_rule_decrease = min_rule_decrease
        self.readability_tradeoff = readability_tradeoff
        self.patience = patience
        self.params = None
        self.normalization = normalization
        self.discretization = discretization

    def search(self):
        raise NotImplementedError

    def _best(self, params):
        param_dict = {self._score(t): t for t in params}
        min_param = min(param_dict)
        return min_param, param_dict[min_param]

    def _score(self, param):
        return param[0] * np.ceil(param[1] * self.readability_tradeoff)

    def _best_param(self, param):
        param_dict = {t[param]: t for t in self.params}
        min_param = min(param_dict)
        return min_param, param_dict[min_param]

    def get_best(self):
        names = ["Combined", "Predictive loss", "N rules"]
        params = [self._best(self.params), self._best_param(0), self._best_param(1)]
        for n, p in zip(names, params):
            self._print_params(n, p[1])
            print()
        return self._best(self.params)[1], self._best_param(0)[1], self._best_param(1)[1]

    def _print_params(self, n, param):
        raise NotImplementedError


class SKEOptimizer(Optimizer, ABC):
    def __init__(self, predictor, dataframe: pd.DataFrame, max_error_increase: float = 1.2,
                 min_rule_decrease: float = 0.9, readability_tradeoff: float = 0.1, patience: int = 5,
                 objective: Objective = Objective.MODEL, output: Target = Target.CONSTANT,
                 normalization=None, discretization=None):
        super().__init__(dataframe, output, max_error_increase, min_rule_decrease, readability_tradeoff,
                         patience, normalization, discretization)
        self.predictor = predictor
        self.objective = objective


class IterativeOptimizer(Optimizer, ABC):
    def __init__(self, dataframe: pd.DataFrame, max_error_increase: float = 1.2,
                 min_rule_decrease: float = 0.9, readability_tradeoff: float = 0.1, max_depth: int = 10,
                 patience: int = 5, output: Target = Target.CONSTANT, normalization=None, discretization=None):
        super().__init__(dataframe, output, max_error_increase, min_rule_decrease, readability_tradeoff,
                         patience, normalization, discretization)
        self.max_depth = max_depth

    def _iteration_improvement(self, best, other):
        if other[0] == best[0]:
            return (best[1] - other[1]) * 2
        return 1 / (
                (1 - other[0] / best[0]) ** self.readability_tradeoff *
                np.ceil(other[1] / self.readability_tradeoff) / np.ceil(best[1] / self.readability_tradeoff)
        )

    def _check_iteration_improvement(self, best, current):
        improvement = \
            self._iteration_improvement([best[0], best[1]], [current[0], current[1]]) if best is not None else np.inf
        if isinstance(improvement, complex):
            improvement = 1.0
        return current, improvement < 1.2
