import numpy as np
import pandas as pd
from enum import Enum
from psyke import Extractor
from psyke.extraction.hypercubic import Grid, FeatureRanker
from psyke.extraction.hypercubic.strategy import AdaptiveStrategy, FixedStrategy
from psyke.tuning import Objective, Optimizer


class PEDRO(Optimizer):
    class Algorithm(Enum):
        GRIDEX = 1,
        GRIDREX = 2

    def __init__(self, predictor, dataframe: pd.DataFrame, max_mae_increase: float = 1.2,
                 min_rule_decrease: float = 0.9, readability_tradeoff: float = 0.1, max_depth: int = 3,
                 patience: int = 3, algorithm: Algorithm = Algorithm.GRIDREX, objective: Objective = Objective.MODEL,
                 normalization=None):
        super().__init__(predictor, algorithm, dataframe, max_mae_increase, min_rule_decrease, readability_tradeoff,
                         max_depth, patience, objective, normalization)
        self.ranked = FeatureRanker(dataframe.columns[:-1]).fit(predictor, dataframe.iloc[:, :-1]).rankings()

    def __search_threshold(self, grid, critical, max_partitions):
        step = self.model_mae / 2.0
        threshold = self.model_mae * 0.5
        params = []
        patience = self.patience
        while patience > 0:
            print("{}. {}. Threshold = {:.2f}. ".format(self.algorithm, grid, threshold), end="")
            extractor = Extractor.gridrex(self.predictor, grid, threshold=threshold, normalization=self.normalization) \
                if self.algorithm == PEDRO.Algorithm.GRIDREX \
                else Extractor.gridex(self.predictor, grid, threshold=threshold, normalization=self.normalization)
            _ = extractor.extract(self.dataframe)
            mae, n = (extractor.mae(self.dataframe, self.predictor) if self.objective == Objective.MODEL else
                      extractor.mae(self.dataframe)), extractor.n_rules
            print("MAE = {:.2f}, {} rules".format(mae, n))

            if len(params) == 0:
                params.append((mae, n, threshold, grid))
                threshold += step
                continue

            if n > max_partitions:
                break

            if n == 1:
                params.append((mae, n, threshold, grid))
                break

            if mae > params[0][0] * self.max_mae_increase:
                break

            improvement = (params[-1][0] / mae) + (1 - n / params[-1][1])

            if improvement <= 1 or n > np.ceil(params[-1][1] * self.min_rule_decrease):
                patience -= 1
                step = max(step, abs(mae - threshold) / max(patience, 1))
            elif not critical:
                patience = self.patience
            if mae != params[-1][0] or n != params[-1][1]:
                params.append((mae, n, threshold, grid))
            threshold += step
        return params

    def __search_depth(self, strategy, critical, max_partitions):
        params = []
        best = None

        for iterations in range(self.max_depth):
            grid = Grid(iterations + 1, strategy)
            p = self.__search_threshold(grid, critical, max_partitions)
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

    def __contains(self, strategies, strategy):
        for s in strategies:
            if strategy.equals(s, self.dataframe.columns[:-1]):
                return True
        return False

    def search(self):
        base_strategy = FixedStrategy(2)
        strategies = [base_strategy, FixedStrategy(3)]

        base_partitions = base_strategy.partition_number(self.dataframe.columns[:-1])

        for n in [2, 3, 5, 10]:
            for th in [0.99, 0.75, 0.67, 0.5, 0.3]:
                strategy = AdaptiveStrategy(self.ranked, [(th, n)])
                if strategy.partition_number(self.dataframe.columns[:-1]) < base_partitions * 3 and \
                        not self.__contains(strategies, strategy):
                    strategies.append(strategy)

        for (a, b) in [(0.33, 0.67), (0.25, 0.75), (0.1, 0.9)]:
            strategy = AdaptiveStrategy(self.ranked, [(a, 2), (b, 3)])
            if strategy.partition_number(self.dataframe.columns[:-1]) < base_partitions * 3 and \
                    not self.__contains(strategies, strategy):
                strategies.append(strategy)

        avg = 0.
        for strategy in strategies:
            avg += strategy.partition_number(self.dataframe.columns[:-1])
        avg /= len(strategies)

        params = []
        for strategy in strategies:
            params += self.__search_depth(strategy,
                                          strategy.partition_number(self.dataframe.columns[:-1]) > avg,
                                          base_partitions * 3)
        self.params = params

    def _print_params(self, name, params):
        print("**********************")
        print(f"Best {name}")
        print("**********************")
        print(f"MAE = {params[0]:.2f}, {params[1]} rules")
        print(f"Threshold = {params[2]:.2f}")
        print(f"Iterations = {params[3].iterations}")
        print(f"Strategy = {params[3].strategy}")
