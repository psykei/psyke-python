from enum import Enum

import numpy as np
import pandas as pd

from psyke import Clustering, EvaluableModel
from psyke.tuning import Optimizer, IterativeOptimizer
from psyke.utils import Target


class OrCHiD(IterativeOptimizer):
    class Algorithm(Enum):
        ExACT = 1,
        CREAM = 2

    def __init__(self, dataframe: pd.DataFrame, algorithm, output: Target = Target.CONSTANT,
                 max_error_increase: float = 1.2, min_rule_decrease: float = 0.9, readability_tradeoff: float = 0.1,
                 patience: int = 5, max_depth: int = 10, gauss_components=10, normalization=None, discretization=None):
        super().__init__(dataframe, max_error_increase, min_rule_decrease, readability_tradeoff, max_depth, patience,
                         output, normalization, discretization)
        self.algorithm = algorithm
        self.gauss_components = gauss_components

    def search(self):
        self.params = self.__search_depth()

    def __search_depth(self):
        params, best = [], None

        for depth in range(1, self.max_depth + 1):
            current_params = self.__search_threshold(depth)
            current_best = self._best(current_params)[1]
            print()
            best, to_break = self._check_iteration_improvement(best, current_best)
            params += current_params

            if len(params) > 1 and to_break:
                break
        return params

    def __search_threshold(self, depth):
        step = 1.0
        threshold = 1.0
        params = []
        patience = self.patience
        while patience > 0:
            print(f"{self.algorithm}. Depth: {depth}. Threshold = {threshold:.2f}. "
                  f"Gaussian components = {self.gauss_components}. ", end="")
            clustering = (Clustering.cream if self.algorithm == OrCHiD.Algorithm.CREAM else Clustering.exact)(
                depth=depth, error_threshold=threshold, gauss_components=self.gauss_components, output=self.output
            )
            clustering.fit(self.dataframe)
            task, metric = \
                (EvaluableModel.Task.CLASSIFICATION, EvaluableModel.ClassificationScore.INVERSE_ACCURACY) \
                if self.output == Target.CLASSIFICATION else \
                (EvaluableModel.Task.REGRESSION, EvaluableModel.RegressionScore.MAE)
            p, n = clustering.score(self.dataframe, None, False, False, task=task,
                                    scoring_function=[metric])[metric][0], clustering.n_rules

            print(f"Predictive loss = {p:.2f}, {n} rules")

            if len(params) == 0:
                params.append((p, n, depth, threshold))
                threshold = p / 20
                step = p / self.patience * 0.75
                continue

            if (n == 1) or (p == 0.0):
                params.append((p, n, depth, threshold))
                break

            if p > params[0][0] * self.max_error_increase:
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
