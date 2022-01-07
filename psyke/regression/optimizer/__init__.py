import numpy as np
import pandas as pd

from psyke import Extractor
from psyke.regression import FixedStrategy, Grid, FeatureRanker
from psyke.regression.strategy import AdaptiveStrategy


class PEDRO:
    def __init__(self, predictor, dataframe: pd.DataFrame, max_mae_increase: float = 1.2,
                 min_rule_decrease: float = 0.9, readability_tradeoff: float = 0.1,
                 max_depth: int = 3, patience: int = 3, alg="GridREx", obj="model"):
        self.predictor = predictor
        self.dataframe = dataframe
        self.ranked = FeatureRanker(dataframe.columns[:-1]).fit(predictor, dataframe.iloc[:, :-1]).rankings()
        self.max_mae_increase = max_mae_increase
        self.min_rule_decrease = min_rule_decrease
        self.readability_tradeoff = readability_tradeoff
        self.patience = patience
        self.max_depth = max_depth
        self.alg = alg
        self.objective = obj
        self.params = None
        self.model_mae = abs(self.predictor.predict(dataframe.iloc[:, :-1]).flatten() -
                             dataframe.iloc[:, -1].values).mean()

    @staticmethod
    def __best(params):
        param_dict = {PEDRO.__score(t): t for t in params}
        min_param = min(param_dict)
        return min_param, param_dict[min_param]

    def __best_param(self, param):
        param_dict = {t[param]: t for t in self.params}
        min_param = min(param_dict)
        return min_param, param_dict[min_param]

    @staticmethod
    def __score(param):
        return param[0] * np.ceil(param[1] / 5)

    def __depth_improvement(self, first, second):
        if second[0] == first[0]:
            return (first[1] - second[1]) * 2
        return 1 / (
                (1 - second[0] / first[0]) ** 0.1 *
                np.ceil(second[1] / self.readability_tradeoff) / np.ceil(first[1] / self.readability_tradeoff)
        )

    def __search_threshold(self, grid, critical, max_partitions):
        step = self.model_mae / 2.0
        threshold = self.model_mae * 0.9
        params = []
        patience = self.patience
        while patience > 0:
            print("{}. {}. Threshold = {:.2f}. ".format(self.alg, grid, threshold), end="")
            extractor = Extractor.gridrex(self.predictor, grid, threshold=threshold) if \
                self.alg == "GridREx" else Extractor.gridex(self.predictor, grid, threshold=threshold)
            _ = extractor.extract(self.dataframe)
            mae, n = extractor.mae(self.dataframe), extractor.n_rules
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
            b = PEDRO.__best(p)[1]
            print()
            improvement = self.__depth_improvement(
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

    @staticmethod
    def __print_params(name, params):
        print("**********************")
        print("*****Best {}*****".format(name))
        print("**********************")
        print("MAE = {:.2f}, {} rules".format(params[0], params[1]))
        print("Threshold = {:.2f}".format(params[2]))
        print("Iterations = {}".format(params[3].iterations))
        print("Strategy = {}".format(params[3].strategy))

    def get_best(self):
        names = [self.alg, "  MAE  ", "N rules"]
        params = [PEDRO.__best(self.params), self.__best_param(0), self.__best_param(1)]
        for n, p in zip(names, params):
            PEDRO.__print_params(n, p[1])
            print()
        return PEDRO.__best(self.params)[1], self.__best_param(0)[1], self.__best_param(1)[1]
