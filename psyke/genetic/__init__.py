from statistics import mode
import numpy as np
from sklearn.linear_model import LinearRegression

from psyke import Target


def regions_from_cuts(x, cuts, features):
    indices = [np.searchsorted(np.array(cut), x[f].to_numpy(), side='right')
               for cut, f in zip(cuts, features)]

    regions = np.zeros(len(x), dtype=int)
    multiplier = 1
    for idx, n in zip(reversed(indices), reversed([len(cut) + 1 for cut in cuts])):
        regions += idx * multiplier
        multiplier *= n
    return regions

def output_estimation(x, y, output, poly, mask, to_pred=None):
    if output == Target.REGRESSION:
        lr = LinearRegression().fit(poly.fit_transform(x)[mask], y[mask])
        return lr if to_pred is None else lr.predict(poly.fit_transform(to_pred))
    if output == Target.CONSTANT:
        return np.mean(y[mask])
    if output == Target.CLASSIFICATION:
        return mode(y[mask])
    raise ValueError('Supported outputs are Target.{REGRESSION, CONSTANT, CLASSIFICATION}')
