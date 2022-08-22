from enum import Enum
import numpy as np
import pandas as pd

from psyke.utils import Target


class Objective(Enum):
    MODEL = 1,
    DATA = 2


class Optimizer:
    def __init__(self, predictor, algorithm, dataframe: pd.DataFrame, max_mae_increase: float = 1.2,
                 min_rule_decrease: float = 0.9, readability_tradeoff: float = 0.1, max_depth: int = 10,
                 patience: int = 5, objective: Objective = Objective.MODEL, normalization=None):
        self.predictor = predictor
        self.algorithm = algorithm
        self.dataframe = dataframe
        self.max_mae_increase = max_mae_increase
        self.min_rule_decrease = min_rule_decrease
        self.readability_tradeoff = readability_tradeoff
        self.patience = patience
        self.max_depth = max_depth
        self.objective = objective
        self.model_mae = abs(self.predictor.predict(dataframe.iloc[:, :-1]).flatten() -
                             self.dataframe.iloc[:, -1].values).mean()
        self.params = None
        self.normalization = normalization

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
