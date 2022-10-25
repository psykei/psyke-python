
import numpy as np
import pandas as pd

from psyke.clustering.orbit import ORBIt
from psyke.tuning.crash import CRASH
from sklearn.metrics import accuracy_score
from kneed import KneeLocator


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
        self.max_time = max_time
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
