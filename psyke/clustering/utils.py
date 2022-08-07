import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


def select_gaussian_mixture(data: pd.DataFrame, max_components) -> tuple[float, int, GaussianMixture]:
    components = range(2, max_components + 1)
    try:
        models = [GaussianMixture(n_components=n).fit(data) for n in components if n <= len(data)]
    except ValueError:
        print(data)
        print(len(data))
    return min([(m.bic(data) / (i + 2), (i + 2), m) for i, m in enumerate(models)])


def select_dbscan_epsilon(data: pd.DataFrame, clusters: int) -> float:
    neighbors = NearestNeighbors(n_neighbors=min(len(data.columns) * 2, len(data))).fit(data)
    distances = sorted(np.mean(neighbors.kneighbors(data)[1], axis=1), reverse=True)
    try:
        kn = KneeLocator([d for d in range(len(distances))], distances,
                         curve='convex', direction='decreasing', online=True)
        if kn.knee is None:
            epsilon = max(distances[-1], 1e-3)
        else:
            epsilon = kn.knee_y
    except (RuntimeWarning, UserWarning, ValueError):
        epsilon = max(distances[-1], 1e-3)
    k = 1.
    dbscan_pred = DBSCAN(eps=epsilon * k).fit_predict(data.iloc[:, :-1])
    # while Counter(dbscan_pred).most_common(1)[0][0] == -1:
    for i in range(1000):
        if len(np.unique(dbscan_pred)) < clusters + 1:
            break
        k += .1
        dbscan_pred = DBSCAN(eps=epsilon * k).fit_predict(data.iloc[:, :-1])
    return epsilon * k
