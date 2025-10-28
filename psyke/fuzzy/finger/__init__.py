import itertools
from collections.abc import Iterable

import numpy as np
import pandas as pd
from deap import base, creator
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from psyke import Target
from psyke.fuzzy import fuzzify, plot_membership, fuzzy_labels, generate_fuzzy_rules, get_activations
from psyke.genetic import regions_from_cuts, output_estimation
from psyke.genetic.fgin import FGIn


class FInGER:

    def __init__(self, predictor, features, sigmas, max_slices, min_rules=1, max_poly=1, alpha=0.5, indpb=0.5,
                 tournsize=3, n_gen=50, n_pop=50, membership_shape='trap', metric='R2', valid=None,
                 output=Target.REGRESSION):

        self.predictor = predictor
        self.features = features
        self.max_features = len(features)
        self.sigmas = sigmas
        self.max_slices = max_slices
        self.min_rules = min_rules
        self.poly = max_poly
        self._output = Target.CLASSIFICATION if isinstance(predictor, ClassifierMixin) else output
        self.valid = valid
        self.trained_poly = None

        self.alpha = alpha
        self.indpb = indpb
        self.tournsize = tournsize
        self.metric = metric
        self.n_gen = n_gen
        self.n_pop = n_pop

        self.shape = membership_shape
        self.valid_masks = None
        self.outputs = None
        self.functions_domains = {}

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

    # TODO: a class for methods and attributes supporting polynomial combinations
    def __poly_names(self):
        return [''.join(['' if pp == 0 else f'{n} * ' if pp == 1 else f'{n}**{pp} * '
                         for pp, n in zip(p, self.trained_poly.feature_names_in_)])[:-3]
                for p in self.trained_poly.powers_]

    @staticmethod
    def _get_cuts(individual, slices):
        boundaries = np.cumsum([0] + list(slices))
        return [sorted(individual[boundaries[i]:boundaries[i + 1]]) for i in range(len(slices))]

    def extract(self, dataframe: pd.DataFrame) -> str:
        best = {}
        for poly in range(self.poly):
            for slices in list(itertools.product(range(1, self.max_slices + 1), repeat=self.max_features)):
                gr = FGIn((dataframe.iloc[:, :-1], dataframe.iloc[:, -1]), self.valid, self.features, self.sigmas,
                          slices, min_rules=self.min_rules, poly=poly + 1, alpha=self.alpha, indpb=self.indpb,
                          tournsize=self.tournsize, membership_shape=self.shape, metric=self.metric,
                          output=self._output, warm=True)

                b, score, _, _ = gr.run(n_gen=self.n_gen, n_pop=self.n_pop)
                best[(score, poly + 1, slices)] = b
        m = min(best)
        poly, slices, best = m[1], m[2], best[m]
        self.trained_poly = PolynomialFeatures(degree=poly, include_bias=False)

        cuts = FInGER._get_cuts(best, slices)
        self.functions_domains = fuzzify(cuts, dataframe.iloc[:, :-1], self.features,
                                         {f: i for i, f in enumerate(dataframe.columns[:-1])}, self.shape)

        masks = np.array([regions_from_cuts(dataframe, cuts, self.features) == r
                          for r in range(np.prod([s + 1 for s in slices]))])
        self.valid_masks = masks.sum(axis=1) >= 3
        masks = masks[self.valid_masks]

        self.outputs = np.array([output_estimation(dataframe.iloc[:, :-1], dataframe.iloc[:, -1], self._output,
                                                   self.trained_poly, mask) for mask in masks]).T

        functions_domains = {k: (v[0], v[1], fuzzy_labels(len(v[0]))) for k, v in self.functions_domains.items()}
        return "\n".join(generate_fuzzy_rules({k: v[2] for k, v in functions_domains.items()}, self.outputs,
                                              dataframe.columns[:-1], self.valid_masks))

    def show_membership_functions(self):
        functions_domains = {k: (v[0], v[1], fuzzy_labels(len(v[0]))) for k, v in self.functions_domains.items()}
        plot_membership(functions_domains)

    def predict(self, dataframe: pd.DataFrame) -> Iterable:
        activations = np.array([get_activations(x, self.functions_domains, self.valid_masks)
                                for _, x in dataframe.iterrows()])

        if self._output == Target.CLASSIFICATION:
            classes, idx = np.unique(self.outputs, return_inverse=True)
            pred = classes[np.argmax(np.vstack([activations[:, idx == i].sum(axis=1) for i, c in enumerate(classes)]),
                                     axis=0)]
        else:
            outputs = self.outputs if self._output == Target.CONSTANT else \
                np.vstack([lr.predict(self.trained_poly.fit_transform(dataframe)) for lr in self.outputs]).T
            pred = (outputs * activations).sum(axis=1)
        return np.array(pred)

    @property
    def n_rules(self):
        return len(self.outputs)
