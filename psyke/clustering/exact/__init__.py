from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from psyke.clustering import InterpretableClustering
from psyke.extraction.hypercubic import Node, ClosedCube, HyperCube
from psyke.clustering.utils import select_gaussian_mixture, select_dbscan_epsilon
from psyke.utils import Target


class ExACT(InterpretableClustering):
    """
    Explanator implementing ExACT algorithm.
    """

    def __init__(self, depth: int, error_threshold: float, output: Target = Target.CONSTANT, gauss_components: int = 5):
        super().__init__(depth, error_threshold, output, gauss_components)
        self._predictor = KNeighborsClassifier() if output == Target.CLASSIFICATION else KNeighborsRegressor()
        self._predictor.n_neighbors = 1

    def __eligible_cubes(self, gauss_pred: np.ndarray, node: Node, clusters: int):
        cubes = []
        for i in range(len(np.unique(gauss_pred))):
            df = node.dataframe.iloc[np.where(gauss_pred == i)]
            if len(df) == 0:
                continue
            cubes.append(self._create_cube(df, clusters))
        indices = [self._indices(cube, node.dataframe) for cube in cubes]
        return cubes, indices

    @staticmethod
    def _indices(cube: ClosedCube, data: pd.DataFrame) -> np.ndarray | None:
        indices = cube.filter_indices(data.iloc[:, :-1])
        if len(data.iloc[indices]) * len(data.iloc[~indices]) == 0:
            return None
        return indices

    def _create_cube(self, dataframe: pd.DataFrame, clusters: int) -> ClosedCube:
        data = ExACT._remove_string_label(dataframe)
        dbscan_pred = DBSCAN(eps=select_dbscan_epsilon(data, clusters)).fit_predict(data.iloc[:, :-1])
        return HyperCube.create_surrounding_cube(
            dataframe.iloc[np.where(dbscan_pred == Counter(dbscan_pred).most_common(1)[0][0])],
            True, self._output
        )

    def extract(self, dataframe: pd.DataFrame) -> Iterable[HyperCube]:
        self._predictor.fit(dataframe.iloc[:, :-1], dataframe.iloc[:, -1])
        self._hypercubes = \
            self._iterate(Node(dataframe, HyperCube.create_surrounding_cube(dataframe, True, self._output)))
        return list(self._hypercubes)

    def print(self):
        for cube in self._hypercubes:
            print(f'Output is {cube.output} if:')
            for feature in cube.dimensions:
                lower, upper = cube[feature]
                print(f'    {feature} is in [{lower:.2f}, {upper:.2f}]')

    @staticmethod
    def _remove_string_label(dataframe: pd.DataFrame):
        return dataframe.replace({dataframe.columns[-1]: {v: k for k, v in dict(
            enumerate(dataframe.iloc[:, -1].unique())
        ).items()}}) if isinstance(dataframe.iloc[0, -1], str) else dataframe

    def _iterate(self, surrounding: Node) -> Iterable[HyperCube]:
        to_split = [(self.error_threshold * 10, 1, 1, surrounding)]
        while len(to_split) > 0:
            to_split.sort(reverse=True)
            (_, depth, _, node) = to_split.pop()
            data = ExACT._remove_string_label(node.dataframe)
            gauss_params = select_gaussian_mixture(data, self.gauss_components)
            gauss_pred = gauss_params[2].predict(data)
            cubes, indices = self.__eligible_cubes(gauss_pred, node, gauss_params[1])
            cubes = [(c.volume(), len(idx), i, idx, c) for i, (c, idx) in enumerate(zip(cubes, indices))
                     if (idx is not None) and (not node.cube.equal(c))]
            if len(cubes) < 1:
                continue
            _, _, _, indices, cube = max(cubes)

            cube.update(node.dataframe[indices], self._predictor)
            node.right = Node(node.dataframe[indices], cube)
            node.cube.update(node.dataframe[~indices], self._predictor)
            node.left = Node(node.dataframe[~indices], node.cube)

            if depth < self.depth and cube.diversity > self.error_threshold:
                to_split.append((cube.diversity, depth + 1, np.random.uniform(), node.right))
        return self._node_to_cubes(surrounding)

    def _node_to_cubes(self, root: Node) -> list[ClosedCube]:
        if root.right is None:
            return [root.cube]
        else:
            return self._node_to_cubes(root.right) + self._node_to_cubes(root.left)

    @property
    def n_rules(self):
        return len(list(self._hypercubes))
