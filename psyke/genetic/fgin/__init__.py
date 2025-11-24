import numpy as np
import pandas as pd

from psyke import Target
from psyke.fuzzy import fuzzify, get_activations
from psyke.genetic import regions_from_cuts, output_estimation
from psyke.genetic.gin import GIn

import skfuzzy as skf


class FGIn(GIn):

    def __init__(self, train, valid, features, sigmas, slices, min_rules=1, poly=1, alpha=0.5, indpb=0.5, tournsize=3,
                 membership_shape='trap', metric='R2', output=Target.REGRESSION, warm=False):
        super().__init__(train, valid, features, sigmas, slices, min_rules, poly, alpha, indpb, tournsize,
                         metric, output, warm)
        self.shape = membership_shape
        self.feature_to_idx = {f: i for i, f in enumerate(self.X.columns)}

    def _evaluate(self, individual=None):
        y_pred, valid_regions = self.__predict(individual or self.best, self.X if self.valid is None else self.valid[0])
        if valid_regions < self.min_rules:
            return -9999,
        return self._score(self.y if self.valid is None else self.valid[1], y_pred),

    def __predict(self, individual=None, to_pred=None):
        cuts = self._get_cuts(individual or self.best)
        masks = np.array([regions_from_cuts(self.X, cuts, self.features) == r
                          for r in range(np.prod([s + 1 for s in self.slices]))])
        valid_masks = masks.sum(axis=1) >= 3
        masks = masks[valid_masks]

        functions_domains = fuzzify(cuts, self.X, self.features, self.feature_to_idx, self.shape)
        pred = np.array([output_estimation(self.X, self.y, self.output, self.poly, mask, to_pred) for mask in masks]).T
        activations = np.array([get_activations(x, functions_domains, valid_masks) for x in to_pred.values])

        if self.output == Target.CLASSIFICATION:
            classes, idx = np.unique(pred, return_inverse=True)
            pred = classes[np.argmax(np.vstack([activations[:, idx == i].sum(axis=1) for i, c in enumerate(classes)]),
                                     axis=0)]
        else:
            pred = (pred * activations).sum(axis=1)

        return pd.DataFrame(pred, index=to_pred.index), len(masks)
