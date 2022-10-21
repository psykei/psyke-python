from enum import Enum

import numpy as np
import pandas as pd

from psyke import Extractor
from psyke.tuning import Objective, Optimizer
from psyke.utils import Target
from psyke.clustering.orbit import ORBIt
from psyke.tuning.crash import CRASH
from sklearn.metrics import accuracy_score


class ORBItTuner:
    def __init__(self, predictor, dataframe: pd.DataFrame, max_mae_increase: float = 1.2,
                 min_rule_decrease: float = 0.9, readability_tradeoff: float = 0.1, max_depth: int = 10,
                 patience: int = 5, normalization=None, gauss_components=10, max_time=1000,
                 min_acc_range=0.1, max_disequation_num=4):
        """

        :param predictor:
        :param dataframe:
        :param max_mae_increase:
        :param min_rule_decrease:
        :param readability_tradeoff:
        :param max_depth:
        :param patience:
        :param normalization:
        :param gauss_components: gauss components used by ORBIt
        :param max_time: maximum amount of time (in seconds) that the algorithm can take, this is NOT the time taken by the tuner,
            but the time used when one execution of ORBIt is done
        :param min_acc_range: min_accuracy_increase will be set between 0 and worse_min_acc
        """
        self.predictor = predictor
        self.dataframe = dataframe
        self.max_mae_increase = max_mae_increase
        self.min_rule_decrease = min_rule_decrease
        self.readability_tradeoff = readability_tradeoff
        self.max_depth = max_depth
        self.max_depth = max_depth
        self.patience = patience
        self.model_mae = abs(self.predictor.predict(dataframe.iloc[:, :-1]).flatten() -
                             self.dataframe.iloc[:, -1].values).mean()
        self.normalization = normalization
        self.gauss_components = gauss_components
        self.max_time=max_time
        self.worse_min_acc = min_acc_range
        self.max_disequation_num = max_disequation_num

        self.hc_tuner = CRASH(predictor=predictor,
                         dataframe=dataframe,
                         max_mae_increase=max_mae_increase,
                         min_rule_decrease=min_rule_decrease,
                         readability_tradeoff=readability_tradeoff,
                         max_depth=max_depth,
                         patience=patience,
                         algorithm=CRASH.Algorithm.ORBIt,
                         normalization=normalization)

        self.depth = 0
        self.threshold = 0
        self.steps = 0
        self.min_accuracy_increase = 0

    def search(self, trials=20):
        self.hc_tuner.search()
        (_, _, depth, threshold) = self.hc_tuner.get_best()[0]
        self.depth = depth
        self.threshold = threshold
        self.steps = self.get_steps(self.max_time)
        self.min_accuracy_increase = self._get_min_acc_increase(trials=trials)
        return self.depth, self.threshold, self.steps, self.min_accuracy_increase

    def get_params(self):
        return self.depth, self.threshold, self.steps, self.min_accuracy_increase

    def get_steps(self, max_time: int) -> int:
        import time
        testing_steps = 10
        orbit = ORBIt(predictor=self.predictor,
                      depth=self.depth,
                      error_threshold=self.threshold,
                      gauss_components=self.gauss_components,
                      normalization=self.normalization,
                      steps=testing_steps,
                      min_accuracy_increase=-1)

        start = time.time()
        _ = orbit.extract(self.dataframe)
        time_used = time.time() - start
        return int(max_time / time_used * testing_steps)

    def _get_min_acc_increase(self, trials=20):
        from kneed import KneeLocator
        df_x = self.dataframe.iloc[:, :-1]
        df_y = self.dataframe.iloc[:, -1]
        starting_min_acc_incr = self.worse_min_acc

        acc_values = np.flip(np.linspace(0, starting_min_acc_incr, trials))
        acc_dict = {}
        print(f"tuning min_accuracy_increase ({trials} trials...")
        fraction_of_trials = int(trials/10)
        for i, min_acc in enumerate(acc_values):
            if i % fraction_of_trials == 0:
                print(f"{round(i / trials * 100, 2)}% done...")
            orbit = ORBIt(predictor=self.predictor,
                          depth=self.depth,
                          error_threshold=self.threshold,
                          gauss_components=self.gauss_components,
                          normalization=self.normalization,
                          steps=self.steps,
                          min_accuracy_increase=min_acc)
            orbit.extract(self.dataframe)
            p_orbit = orbit.predict(df_x)
            acc_dict[orbit.n_oblique_rules] = (min_acc, accuracy_score(df_y, p_orbit))
        x = []
        y = []
        for x_ in acc_dict:
            _, model_acc = acc_dict[x_]
            x.append(x_)
            y.append(model_acc)
        if len(acc_dict) < 3:
            return 0

        kneedle = KneeLocator(x, y, S=1, curve="concave", direction="increasing", online=True)
        from matplotlib import pyplot as plt
        plt.plot(x, y)
        plt.show()
        if kneedle.knee is None:
            return starting_min_acc_incr
        else:
            min_acc, _ = acc_dict[kneedle.knee]
            return min_acc

#
#
# class HCTuner:
#     """
#     same algorithm of CRASH, used to find optimal depth and threshold for ORBIt when only hyper-cubes are considered
#     """
#
#     def __init__(self, predictor, dataframe: pd.DataFrame, max_mae_increase: float = 1.2,
#                  min_rule_decrease: float = 0.9, readability_tradeoff: float = 0.1, max_depth: int = 10,
#                  patience: int = 5, normalization=None, gauss_components=10):
#         self.predictor = predictor
#         self.dataframe = dataframe
#         self.max_mae_increase = max_mae_increase
#         self.min_rule_decrease = min_rule_decrease
#         self.readability_tradeoff = readability_tradeoff
#         self.max_depth = max_depth
#         self.max_depth = max_depth
#         self.patience = patience
#         self.model_mae = abs(self.predictor.predict(dataframe.iloc[:, :-1]).flatten() -
#                              self.dataframe.iloc[:, -1].values).mean()
#         self.normalization = normalization
#         self.gauss_components = gauss_components
#
#     def search(self):
#         self.params = self.__search_depth()
#         print("finished")
#
#     def __search_depth(self):
#         params = []
#         best = None
#
#         for depth in range(1, self.max_depth + 1):
#             p = self.__search_threshold(depth)
#             b = Optimizer._best(p)[1]
#             print()
#             improvement = self._depth_improvement(
#                 [best[0], best[1]], [b[0], b[1]]
#             ) if best is not None else np.inf
#
#             best = b
#             params += p
#
#             if len(params) > 1 and improvement < 1.2:
#                 break
#         return params
#
#     def __search_threshold(self, depth):
#         step = self.model_mae / 2.0
#         threshold = self.model_mae * 0.9
#         params = []
#         patience = self.patience
#         while patience > 0:
#             print(f"Depth: {depth}. Threshold = {threshold:.2f}. ", end="")
#             min_accuracy_increase = 0.01
#             extractor = Extractor.orbit(self.predictor, depth, threshold, self.gauss_components,
#                                         normalization=self.normalization, steps=0,
#                                         min_accuracy_increase=min_accuracy_increase)
#             assert isinstance(extractor, ORBIt)
#             _ = extractor.extract(self.dataframe)
#             mae, n = extractor.mae(self.dataframe, self.predictor), extractor.n_rules
#             print(f"MAE = {mae:.2f}, {n} rules")
#
#             if len(params) == 0:
#                 params.append((mae, n, depth, threshold))
#                 threshold += step
#                 continue
#
#             if (n == 1) or (mae == 0.0):
#                 params.append((mae, n, depth, threshold))
#                 break
#
#             if mae > params[0][0] * self.max_mae_increase:
#                 break
#
#             improvement = (params[-1][0] / mae) + (1 - n / params[-1][1])
#
#             if improvement <= 1 or n > np.ceil(params[-1][1] * self.min_rule_decrease):
#                 patience -= 1
#                 step = max(step, abs(mae - threshold) / max(patience, 1))
#             if mae != params[-1][0] or n != params[-1][1]:
#                 params.append((mae, n, depth, threshold))
#             threshold += step
#         return params
#
#     def _depth_improvement(self, first, second):
#         if second[0] == first[0]:
#             return (first[1] - second[1]) * 2
#         return 1 / (
#                 (1 - second[0] / first[0]) ** self.readability_tradeoff *
#                 np.ceil(second[1] / self.readability_tradeoff) / np.ceil(first[1] / self.readability_tradeoff)
#         )
#
#     def _get_step_num(self, time: int):
#         (_, _, depth, threshold) = self.get_best()[0]
#
#     def _best_param(self, param):
#         param_dict = {t[param]: t for t in self.params}
#         min_param = min(param_dict)
#         return min_param, param_dict[min_param]
#
#     def get_best(self):
#         names = ["ORBIt", "  MAE  ", "N rules"]
#         params = [Optimizer._best(self.params), self._best_param(0), self._best_param(1)]
#         for n, p in zip(names, params):
#             self._print_params(n, p[1])
#             print()
#         return Optimizer._best(self.params)[1], self._best_param(0)[1], self._best_param(1)[1]
#
#     def _print_params(self, name, params):
#         print("**********************")
#         print(f"Best {name}")
#         print("**********************")
#         print(f"MAE = {params[0]:.2f}, {params[1]} rules")
#         print(f"Threshold = {params[3]:.2f}")
#         print(f"Depth = {params[2]}")
