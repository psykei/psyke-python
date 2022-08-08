from enum import Enum
import numpy as np


class Objective(Enum):
    MODEL = 1,
    DATA = 2


class Optimizer:
    def __init__(self, readability_tradeoff, algorithm, params=None):
        self.readability_tradeoff = readability_tradeoff
        self.algorithm = algorithm
        self.params = params

    def _depth_improvement(self, first, second):
        if second[0] == first[0]:
            return (first[1] - second[1]) * 2
        return 1 / (
                (1 - second[0] / first[0]) ** self.readability_tradeoff *
                np.ceil(second[1] / self.readability_tradeoff) / np.ceil(first[1] / self.readability_tradeoff)
        )

    @staticmethod
    def _best(params):
        param_dict = {Optimizer.__score(t): t for t in params}
        min_param = min(param_dict)
        return min_param, param_dict[min_param]

    @staticmethod
    def __score(param):
        return param[0] * np.ceil(param[1] / 5)

    def _best_param(self, param):
        param_dict = {t[param]: t for t in self.params}
        min_param = min(param_dict)
        return min_param, param_dict[min_param]

    def get_best(self):
        names = [self.algorithm, "  MAE  ", "N rules"]
        params = [Optimizer._best(self.params), self._best_param(0), self._best_param(1)]
        for n, p in zip(names, params):
            self._print_params(n, p[1])
            print()
        return Optimizer._best(self.params)[1], self._best_param(0)[1], self._best_param(1)[1]

    def _print_params(self, n, param):
        raise NotImplementedError
