import itertools
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import PolynomialFeatures
from tuprolog.theory import Theory

from psyke import get_default_random_seed, Target
from psyke.extraction.hypercubic import HyperCubeExtractor, HyperCube, RegressionCube

from deap import base, creator

from psyke.genetic.gin import GIn


class GInGER(HyperCubeExtractor):
    """
    Explanator implementing GInGER algorithm.
    """

    def __init__(self, predictor, features, sigmas, max_slices, min_rules=1, max_poly=1, alpha=0.5, indpb=0.5,
                 tournsize=3, metric='R2', n_gen=50, n_pop=50, threshold=None, valid=None,
                 output: Target = Target.REGRESSION, normalization=None, seed: int = get_default_random_seed()):
        super().__init__(predictor, output=Target.CLASSIFICATION if isinstance(predictor, ClassifierMixin) else output,
                         normalization=normalization)
        self.threshold = threshold
        np.random.seed(seed)

        self.features = features
        self.max_features = len(features)
        self.sigmas = sigmas
        self.max_slices = max_slices
        self.min_rules = min_rules
        self.poly = max_poly
        self.trained_poly = None

        self.alpha = alpha
        self.indpb = indpb
        self.tournsize = tournsize
        self.metric = metric

        self.n_gen = n_gen
        self.n_pop = n_pop
        self.valid = valid

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

    def __poly_names(self):
        return [''.join(['' if pp == 0 else f'{n} * ' if pp == 1 else f'{n}**{pp} * '
                         for pp, n in zip(p, self.trained_poly.feature_names_in_)])[:-3]
                for p in self.trained_poly.powers_]

    def _predict(self, dataframe: pd.DataFrame) -> Iterable:
        dataframe = pd.DataFrame(self.trained_poly.fit_transform(dataframe), columns=self.__poly_names())
        return np.array([self._predict_from_cubes(row.to_dict()) for _, row in dataframe.iterrows()])

    def _extract(self, dataframe: pd.DataFrame) -> Theory:
        best = {}
        for poly in range(self.poly):
            for slices in list(itertools.product(range(1, self.max_slices + 1), repeat=self.max_features)):
                gr = GIn((dataframe.iloc[:, :-1], dataframe.iloc[:, -1]), self.valid, self.features, self.sigmas,
                         slices, min_rules=self.min_rules, poly=poly + 1, alpha=self.alpha, indpb=self.indpb,
                         tournsize=self.tournsize, metric=self.metric, output=self._output, warm=True)

                b, score, _, _ = gr.run(n_gen=self.n_gen, n_pop=self.n_pop)
                best[(score, poly + 1, slices)] = b
        m = min(best)
        poly, slices, best = m[1], m[2], best[m]
        self.trained_poly = PolynomialFeatures(degree=poly, include_bias=False)
        transformed = pd.DataFrame(self.trained_poly.fit_transform(dataframe.iloc[:, :-1]), columns=self.__poly_names())
        transformed[dataframe.columns[-1]] = dataframe.iloc[:, -1].values

        self._surrounding = HyperCube.create_surrounding_cube(transformed, output=self._output)

        cuts = [sorted(best[sum(slices[:i]):sum(slices[:i + 1])]) for i in range(len(slices))]

        intervals = [[(transformed[self.features[i]].min(), cut[0])] +
                     [(cut[i], cut[i + 1]) for i in range(len(cut) - 1)] +
                     [(cut[-1], transformed[self.features[i]].max())] for i, cut in enumerate(cuts)]

        hypercubes = [{f: iv for f, iv in zip(self.features, combo)} for combo in itertools.product(*intervals)]
        mi_ma = {f: (transformed[f].min(), transformed[f].max()) for f in transformed.columns if f not in self.features}
        self._hypercubes = [self._default_cube({feat: h[feat] if feat in self.features else mi_ma[feat]
                                                for feat in transformed.columns[:-1]}) for h in hypercubes]
        self._hypercubes = [c for c in self._hypercubes if c.count(transformed) >= 2]
        for c in self._hypercubes:
            for feature in transformed.columns:
                if feature not in self.features:
                    for direction in ['+', '-']:
                        c.set_infinite(feature, direction)
            c.update(transformed)
        if self.threshold is not None:
            self._hypercubes = self._merge(self._hypercubes, transformed)
        return self._create_theory(transformed)

    def make_fair(self, features: Iterable[str]):
        self._dimensions_to_ignore.update(features)
