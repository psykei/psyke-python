from enum import Enum

import numpy as np
import pandas as pd

from psyke import Extractor, Clustering
from psyke.tuning import Objective, Optimizer
from psyke.utils import Target


class CRASH(Optimizer):
    class Algorithm(Enum):
        ExACT = 1,
        CREAM = 2

    def __init__(self, predictor, dataframe: pd.DataFrame, max_mae_increase: float = 1.2,
                 min_rule_decrease: float = 0.9, readability_tradeoff: float = 0.1, max_depth: int = 10,
                 patience: int = 5, algorithm: Algorithm = Algorithm.CREAM, output: Target = Target.CONSTANT,
                 objective: Objective = Objective.MODEL, normalization=None):
        super().__init__(predictor, algorithm, dataframe, max_mae_increase, min_rule_decrease, readability_tradeoff,
                         max_depth, patience, objective, normalization)
        self.output = output

    def search(self):
        self.params = self.__search_depth()

    def __search_depth(self):
        params = []
        best = None

        for depth in range(1, self.max_depth + 1):
            p = self.__search_threshold(depth)
            b = Optimizer._best(p)[1]
            print()
            improvement = self._depth_improvement(
                [best[0], best[1]], [b[0], b[1]]
            ) if best is not None else np.inf

            best = b
            params += p

            if len(params) > 1 and improvement < 1.2:
                break
        return params

    def _print_params(self, name, params):
        print("**********************")
        print(f"Best {name}")
        print("**********************")
        print(f"MAE = {params[0]:.2f}, {params[1]} rules")
        print(f"Threshold = {params[3]:.2f}")
        print(f"Depth = {params[2]}")
