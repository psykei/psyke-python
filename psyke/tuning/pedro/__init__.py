import numpy as np
import pandas as pd
from enum import Enum

from sklearn.metrics import accuracy_score

from psyke import Extractor, Target
from psyke.extraction.hypercubic import Grid, FeatureRanker
from psyke.extraction.hypercubic.strategy import AdaptiveStrategy, FixedStrategy
from psyke.tuning import Objective, IterativeOptimizer, SKEOptimizer


class PEDRO(SKEOptimizer, IterativeOptimizer):
    class Algorithm(Enum):
        GRIDEX = 1,
        GRIDREX = 2,
        HEX = 3

    def __init__(self, predictor, dataframe: pd.DataFrame, max_error_increase: float = 1.2,
                 min_rule_decrease: float = 0.9, readability_tradeoff: float = 0.1, max_depth: int = 3,
                 patience: int = 3, algorithm: Algorithm = Algorithm.GRIDREX, objective: Objective = Objective.MODEL,
                 output: Target = Target.CONSTANT, normalization=None, discretization=None):
        SKEOptimizer.__init__(self, predictor, dataframe, max_error_increase, min_rule_decrease,
                              readability_tradeoff, patience, objective, output, normalization, discretization)
        IterativeOptimizer.__init__(self, dataframe, max_error_increase, min_rule_decrease, readability_tradeoff,
                                    max_depth, patience, output, normalization, discretization)
        self.algorithm = Extractor.gridrex if algorithm == PEDRO.Algorithm.GRIDREX else \
            Extractor.gridex if algorithm == PEDRO.Algorithm.GRIDEX else Extractor.hex
        self.algorithm_name = "GridREx" if algorithm == PEDRO.Algorithm.GRIDREX else \
            "GridEx" if algorithm == PEDRO.Algorithm.GRIDEX else "HEx"
        self.ranked = FeatureRanker(dataframe.columns[:-1]).fit(predictor, dataframe.iloc[:, :-1]).rankings()
        predictions = self.predictor.predict(dataframe.iloc[:, :-1]).flatten()
        expected = self.dataframe.iloc[:, -1].values
        self.error = 1 - accuracy_score(predictions, expected) if output == Target.CLASSIFICATION else \
            abs(predictions - expected).mean()

    def _search_depth(self, strategy, critical, max_partitions):
        params, best = [], None

        for iterations in range(self.max_depth):
            current_params = self.__search_threshold(Grid(iterations + 1, strategy), critical, max_partitions)
            current_best = self._best(current_params)[1]
            print()
            best, to_break = self._check_iteration_improvement(best, current_best)
            params += current_params

            if len(params) > 1 and to_break:
                break
        return params

    def __search_threshold(self, grid, critical, max_partitions):
        step = self.error / 2.0
        threshold = self.error * 0.5
        params = []
        patience = self.patience
        while patience > 0:
            print("{}. {}. Threshold = {:.2f}. ".format(self.algorithm_name, grid, threshold), end="")
            param_dict = dict(min_examples=25, threshold=threshold, normalization=self.normalization)
            if self.algorithm != Extractor.gridrex:
                param_dict['output'] = self.output
            extractor = self.algorithm(self.predictor, grid, **param_dict)
            _ = extractor.extract(self.dataframe)
            error_function = (lambda *x: 1 - extractor.accuracy(*x)) if self.output == Target.CLASSIFICATION \
                else extractor.mae
            error, n = (error_function(self.dataframe, self.predictor) if self.objective == Objective.MODEL else
                        error_function(self.dataframe)), extractor.n_rules
            print("MAE = {:.2f}, {} rules".format(error, n))

            if len(params) == 0:
                params.append((error, n, threshold, grid))
                threshold += step
                continue

            if n > max_partitions:
                break

            if n == 1:
                params.append((error, n, threshold, grid))
                break

            if error > params[0][0] * self.max_error_increase:
                break

            improvement = (params[-1][0] / error) + (1 - n / params[-1][1])

            if improvement <= 1 or n > np.ceil(params[-1][1] * self.min_rule_decrease):
                patience -= 1
                step = max(step, abs(error - threshold) / max(patience, 1))
            elif not critical:
                patience = self.patience
            if error != params[-1][0] or n != params[-1][1]:
                params.append((error, n, threshold, grid))
            threshold += step
        return params

    def __contains(self, strategies, strategy):
        for s in strategies:
            if strategy.equals(s, self.dataframe.columns[:-1]):
                return True
        return False

    def search(self):
        max_partitions = 200
        base_partitions = FixedStrategy(2).partition_number(self.dataframe.columns[:-1]) * 3
        if base_partitions <= max_partitions:
            strategies = [FixedStrategy(2)]
            if FixedStrategy(3).partition_number(self.dataframe.columns[:-1]) <= max_partitions:
                strategies.append(FixedStrategy(3))
        else:
            strategies = []
            base_partitions = max_partitions

        for n in [2, 3, 5, 10]:
            for th in [0.99, 0.75, 0.67, 0.5, 0.3]:
                strategy = AdaptiveStrategy(self.ranked, [(th, n)])
                if strategy.partition_number(self.dataframe.columns[:-1]) < base_partitions and \
                        not self.__contains(strategies, strategy):
                    strategies.append(strategy)

        for (a, b) in [(0.33, 0.67), (0.25, 0.75), (0.1, 0.9)]:
            strategy = AdaptiveStrategy(self.ranked, [(a, 2), (b, 3)])
            if strategy.partition_number(self.dataframe.columns[:-1]) < base_partitions and \
                    not self.__contains(strategies, strategy):
                strategies.append(strategy)

        avg = 0.
        for strategy in strategies:
            avg += strategy.partition_number(self.dataframe.columns[:-1])
        avg /= len(strategies)

        params = []
        for strategy in strategies:
            params += self._search_depth(strategy,
                                         strategy.partition_number(self.dataframe.columns[:-1]) > avg,
                                         base_partitions)
        self.params = params

    def _print_params(self, name, params):
        print("**********************")
        print(f"Best {name}")
        print("**********************")
        print(f"Error = {params[0]:.2f}, {params[1]} rules")
        print(f"Threshold = {params[2]:.2f}")
        print(f"Iterations = {params[3].iterations}")
        print(f"Strategy = {params[3].strategy}")
