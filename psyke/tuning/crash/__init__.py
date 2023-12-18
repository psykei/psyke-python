from enum import Enum

import numpy as np
import pandas as pd

from psyke.tuning import Objective, Optimizer, SKEOptimizer
from psyke.tuning.orchid import OrCHiD
from psyke.utils import Target


class CRASH(SKEOptimizer):
    def __init__(self, predictor, dataframe: pd.DataFrame, max_error_increase: float = 1.2,
                 min_rule_decrease: float = 0.9, readability_tradeoff: float = 0.1, max_depth: int = 10,
                 max_gauss_components: int = 10, patience: int = 5, output: Target = Target.CONSTANT,
                 objective: Objective = Objective.MODEL, normalization=None, discretization=None):
        super().__init__(predictor, dataframe, max_error_increase, min_rule_decrease, readability_tradeoff,
                         patience, objective, output, normalization, discretization)
        self.max_depth = max_depth
        self.max_gauss_components = max_gauss_components

    def search(self):
        self.params = []
        for algorithm in [OrCHiD.Algorithm.ExACT, OrCHiD.Algorithm.CREAM]:
            self.params += self.__search_algorithm(algorithm)

    def __search_algorithm(self, algorithm):
        params = []
        best = None

        for gauss_components in range(2, self.max_gauss_components + 1):
            data = self.dataframe.sample(n=gauss_components * 100) if gauss_components * 100 < len(self.dataframe) \
                else self.dataframe
        return params

    def _print_params(self, name, params):
        print("**********************")
        print(f"Best {name}")
        print("**********************")
        print(f"MAE = {params[0]:.2f}, {params[1]} rules")
        print(f"Threshold = {params[3]:.2f}")
        print(f"Depth = {params[2]}")
