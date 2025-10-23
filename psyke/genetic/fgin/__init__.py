import numpy as np
import pandas as pd

from psyke import Target
from psyke.genetic.gin import GIn

import skfuzzy as skf


class FGIn(GIn):

    def __init__(self, train, valid, features, sigmas, slices, min_rules=1, poly=1, alpha=0.5, indpb=0.5, tournsize=3,
                 metric='R2', output=Target.REGRESSION, warm=False):
        super().__init__(train, valid, features, sigmas, slices, min_rules, poly, alpha, indpb, tournsize,
                         metric, output, warm)
        self.feature_to_idx = {f: i for i, f in enumerate(self.X.columns)}

    def _evaluate(self, individual=None):
        y_pred, valid_regions = self.__predict(individual or self.best, self.X if self.valid is None else self.valid[0])
        if valid_regions < self.min_rules:
            return -9999,
        return self._score(self.y if self.valid is None else self.valid[1], y_pred),

    @staticmethod
    def __generate_membership(var, domain, thresholds, shape='tri'):
        th = [var.min()] + [min(max(t, var.min()), var.max()) for t in thresholds] + [var.max()]

        if shape == 'tri':
            mid = [(x1 + x2) / 2 for x1, x2 in zip(th[:-1], th[1:])]
            return [skf.trapmf(domain, [domain.min()] * 2 + mid[:2])] + \
                   [skf.trimf(domain, [x1, x2, x3]) for x1, x2, x3 in zip(mid[:-2], mid[1:-1], mid[2:])] + \
                   [skf.trapmf(domain, mid[-2:] + [domain.max()] * 2)]
        if shape == 'trap':
            beg = [None, domain.min()] + [(3 * x1 + x2) / 4 for x1, x2 in zip(th[1:-1], th[2:])] + [domain.max()]
            end = [domain.min()] + [(x1 + 3 * x2) / 4 for x1, x2 in zip(th[:-2], th[1:-1])] + [domain.max()]
            return [skf.trapmf(domain, [end[i - 1], beg[i], end[i], beg[i + 1]]) for i in range(1, len(th))]
        raise ValueError('Supported shape values are only \'tri\' and \'trap\'')

    @staticmethod
    def __extend_domain(x, q_low=0.05, q_high=0.95, p=0.05, k_sigma=2.0, abs_min_margin=0.0):
        ql, qh = np.quantile(x, [q_low, q_high])
        margin = max(p * (qh - ql), k_sigma * np.std(x), abs_min_margin)
        return np.array([ql - margin, qh + margin])

    def __get_activations(self, x, functions_domains, valid_masks):
        levels = [np.array([skf.interp_membership(domain, mf, x[index]) for mf in mfs])
                  for mfs, domain, index in functions_domains.values()]
        return np.prod(np.meshgrid(*levels, indexing='ij'), axis=0).ravel()[valid_masks]

    def __fuzzify(self, cuts):
        cuts = dict(zip(self.features, cuts))
        doms = {c: FGIn.__extend_domain(self.X[c]) for c in self.features}
        return {c: (FGIn.__generate_membership(self.X[c], doms[c], cuts[c], 'trap'), doms[c],
                    self.feature_to_idx[c]) for c in self.features}

    def __predict(self, individual=None, to_pred=None):
        cuts = self._get_cuts(individual or self.best)
        masks = np.array([self._region(to_pred, cuts) == r for r in range(np.prod([s + 1 for s in self.slices]))])
        valid_masks = masks.sum(axis=1) >= 3

        masks = [mask for mask in masks if mask.sum() >= 3]
        functions_domains = self.__fuzzify(cuts)

        pred = np.array([self._output_estimation(mask, to_pred) for mask in masks]).T
        activations = np.array([self.__get_activations(x, functions_domains, valid_masks) for x in to_pred.values])

        if self.output == Target.CLASSIFICATION:
            classes, idx = np.unique(pred, return_inverse=True)
            pred = classes[np.argmax(np.vstack([activations[:, idx == i].sum(axis=1) for i, c in enumerate(classes)]),
                                     axis=0)]
        else:
            pred = (pred * activations).sum(axis=1)

        return pd.DataFrame(pred, index=to_pred.index), len(masks)
