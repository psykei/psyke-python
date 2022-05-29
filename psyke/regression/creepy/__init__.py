from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from tuprolog.theory import Theory
from psyke.regression import ClusterExtractor, Node, ClosedCube, HyperCubeExtractor, HyperCube
from psyke.regression.utils import select_gaussian_mixture, select_dbscan_epsilon


class CReEPy(ClusterExtractor):
    """
    Explanator implementing CReEPy algorithm.
    """

    def __init__(self, predictor, depth: int, error_threshold: float,
                 gauss_components: int = 5, constant: bool = False):
        super().__init__(predictor, depth, error_threshold, gauss_components, constant)

    def _predict(self, data: dict[str, float]) -> float:
        data = {k: v for k, v in data.items()}
        return HyperCubeExtractor._get_cube_output(self._hypercubes.search(data), data)

    def _split(self, right: ClosedCube, outer_cube: ClosedCube, data: pd.DataFrame, indices: ndarray):
        right.update(data.iloc[indices], self.predictor)
        left = outer_cube.copy()
        left.update(data.iloc[~indices], self.predictor)
        return right, left

    def __eligible_cubes(self, gauss_pred: ndarray, node: Node):
        cubes = [
            self._create_cube(node.dataframe.iloc[np.where(gauss_pred == i)]) for i in range(len(np.unique(gauss_pred)))
        ]
        indices = [self._indices(cube, node.dataframe) for cube in cubes]
        return cubes, indices

    @staticmethod
    def _indices(cube: ClosedCube, data: pd.DataFrame) -> ndarray | None:
        indices = cube.filter_indices(data.iloc[:, :-1])
        if len(data.iloc[indices]) * len(data.iloc[~indices]) == 0:
            return None
        return indices

    def _create_cube(self, df: pd.DataFrame) -> ClosedCube:
        dbscan_pred = DBSCAN(eps=select_dbscan_epsilon(df)).fit_predict(df.iloc[:, :-1])
        return HyperCube.create_surrounding_cube(
            df.iloc[np.where(dbscan_pred == Counter(dbscan_pred).most_common(1)[0][0])],
            True, self._constant
        )

    def extract(self, dataframe: pd.DataFrame) -> Theory:
        self._hypercubes = Node(dataframe, HyperCube.create_surrounding_cube(dataframe, True, self._constant))
        self._iterate(self._hypercubes)
        # return self._create_theory(dataframe)
        return None

    def _iterate(self, surrounding: Node) -> None:
        to_split = [(self.error_threshold * 10, 1, surrounding)]
        while len(to_split) > 0:
            to_split.sort(reverse=True)
            (_, depth, node) = to_split.pop()
            gauss_pred = select_gaussian_mixture(node.dataframe, self.gauss_components).predict(node.dataframe)
            cubes, indices = self.__eligible_cubes(gauss_pred, node)
            cubes = [(c.volume(), len(idx), i, idx, c)
                     for i, (c, idx) in enumerate(zip(cubes, indices)) if (idx is not None) and (not node.cube.equal(c))]
            if len(cubes) < 1:
                continue
            _, _, _, indices, cube = max(cubes)

            cube.update(node.dataframe[indices], self.predictor)
            node.right = Node(node.dataframe[indices], cube)
            node.cube.update(node.dataframe[~indices], self.predictor)
            node.left = Node(node.dataframe[~indices], node.cube)

            if depth < self.depth and cube.diversity > self.error_threshold:
                to_split.append((cube.diversity, depth + 1, node.right))

    def _calculate_error(self, dataframe: pd.DataFrame, cube: ClosedCube) -> float:
        output = cube.output
        if isinstance(output, float):
            return abs(self.predictor.predict(dataframe) - output).mean()
        elif isinstance(output, LinearRegression):
            return abs(self.predictor.predict(dataframe) - output.predict(dataframe)).mean()

    @property
    def n_rules(self):
        return self._hypercubes.leaves
