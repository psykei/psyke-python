from __future__ import annotations
import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from tuprolog.theory import Theory
from psyke.regression import ClusterExtractor, Node, ClosedCube, HyperCubeExtractor


class CReEPy(ClusterExtractor):
    """
    Explanator implementing CReEPy algorithm.
    """

    def __init__(self, predictor, depth: int, dbscan_threshold: float, error_threshold: float, constant: bool = False):
        super().__init__(predictor, depth, dbscan_threshold, error_threshold, constant)

    def _predict(self, data: dict[str, float]) -> float:
        data = {k: v for k, v in data.items()}
        return HyperCubeExtractor._get_cube_output(self._hypercubes.search(data), data)

    def __split_with_errors(self, right: ClosedCube, outer_cube: ClosedCube, data: pd.DataFrame, indices: ndarray):
        right.update(data.iloc[indices], self.predictor)
        left = outer_cube.copy()
        left.update(data.iloc[~indices], self.predictor)
        return right, self._calculate_error(data.iloc[indices, :-1], right), \
            left, self._calculate_error(data.iloc[~indices, :-1], left)

    def __eligible_cubes(self, gauss_pred: ndarray, node: Node):
        cubes = [self._create_cube(node.dataframe.iloc[np.where(gauss_pred == i)]) for i in range(2)]
        indices = [self.__indices(cube, node.dataframe) for cube in cubes]
        return cubes, indices

    @staticmethod
    def __indices(cube: ClosedCube, data: pd.DataFrame) -> ndarray | None:
        indices = cube.filter_indices(data.iloc[:, :-1])
        if len(data.iloc[indices]) * len(data.iloc[~indices]) == 0:
            return None
        return indices

    def _iterate(self, surrounding: Node) -> None:
        to_split = [(self.error_threshold * 10, 1, surrounding)]
        while len(to_split) > 0:
            to_split.sort(reverse=True)
            (_, depth, node) = to_split.pop()
            gauss_pred = GaussianMixture(n_components=2).fit_predict(node.dataframe)

            cubes, indices = self.__eligible_cubes(gauss_pred, node)
            cubes = [(c.volume(), len(idx), idx, c) for c, idx in zip(cubes, indices) if idx is not None]

            if len(cubes) < 1:
                continue
            _, _, indices, cube = max(cubes)
            if node.cube.equal(cube):
                _, _, indices, cube = min(cubes)

            cube.update(node.dataframe[indices], self.predictor)
            node.right = Node(node.dataframe[indices], cube)
            node.cube.update(node.dataframe[~indices], self.predictor)
            node.left = Node(node.dataframe[~indices], node.cube)
            error = self._calculate_error(node.dataframe.iloc[indices, :-1], cube)

            if depth < self.depth and error > self.error_threshold:
                to_split.append((error, depth + 1, node.right))

        #    plt.scatter(node.dataframe.X, node.dataframe.Y,
        #                c=gauss_pred, s=.5)
        #    plt.gca().set_aspect('equal')
        #    plt.xlim((0, 1))
        #    plt.ylim((0, 1))
        #    plt.show()

    def _calculate_error(self, dataframe: pd.DataFrame, cube: ClosedCube) -> float:
        output = cube.output
        if isinstance(output, float):
            return abs(self.predictor.predict(dataframe) - output).mean()
        elif isinstance(output, LinearRegression):
            return abs(self.predictor.predict(dataframe) - output.predict(dataframe)).mean()

    @property
    def n_rules(self):
        return self._hypercubes.leaves
