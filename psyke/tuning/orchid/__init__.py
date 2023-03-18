from enum import Enum

import numpy as np
import pandas as pd

from psyke import Clustering, EvaluableModel
from psyke.tuning import Optimizer
from psyke.utils import Target


class OrCHiD(Optimizer):
    class Algorithm(Enum):
        ExACT = 1,
        CREAM = 2

    def __init__(self, dataframe: pd.DataFrame, algorithm, output: Target = Target.CONSTANT,
                 max_mae_increase: float = 1.2, min_rule_decrease: float = 0.9, readability_tradeoff: float = 0.1,
                 patience: int = 5, max_depth: int = 10, normalization=None, discretization=None):
        super().__init__(dataframe, algorithm, output, max_mae_increase, min_rule_decrease, readability_tradeoff,
                         patience, normalization, discretization)
        self.max_depth = max_depth

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

    def __search_threshold(self, depth):
        step = 1.0
        threshold = 1.0  # self.max_mae_increase * 0.9
        params = []
        patience = self.patience
        while patience > 0:
            print(f"{self.algorithm}. Depth: {depth}. Threshold = {threshold:.2f}. ", end="")
            clustering = (Clustering.cream if self.algorithm == OrCHiD.Algorithm.CREAM else Clustering.exact)(
                depth=depth, error_threshold=threshold, gauss_components=10, output=self.output
            )
            clustering.fit(self.dataframe)
            task, metric = \
                (EvaluableModel.Task.CLASSIFICATION, EvaluableModel.ClassificationScore.INVERSE_ACCURACY) \
                if self.output == Target.CLASSIFICATION else \
                (EvaluableModel.Task.REGRESSION, EvaluableModel.RegressionScore.MAE)
            p, n = clustering.score(self.dataframe, None, False, False, task, [metric])[metric][0], clustering.n_rules

            print(f"Predictive loss = {p:.2f}, {n} rules")

            if len(params) == 0:
                params.append((p, n, depth, threshold))
                threshold = p / 20
                step = p / self.patience * 0.75
                continue

            if (n == 1) or (p == 0.0):
                params.append((p, n, depth, threshold))
                break

            if p > params[0][0] * self.max_mae_increase:
                break

            improvement = (params[-1][0] / p) + (1 - n / params[-1][1])

            if improvement <= 1 or n > np.ceil(params[-1][1] * self.min_rule_decrease):
                patience -= 1
            if p != params[-1][0] or n != params[-1][1]:
                params.append((p, n, depth, threshold))
            threshold += step
        return params

    def _print_params(self, name, params):
        print("*" * 40)
        print(f"* Best {name}")
        print("*" * 40)
        print(f"* Predictive loss = {params[0]:.2f}, {params[1]} rules")
        print(f"* Threshold = {params[3]:.2f}")
        print(f"* Depth = {params[2]}")
        print("*" * 40)
