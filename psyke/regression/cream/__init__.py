from collections import Counter
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from tuprolog.theory import Theory
import random as rnd
from psyke import get_default_random_seed
from psyke.regression import ClusterExtractor, HyperCube, Node, ClosedCube, HyperCubeExtractor


class CREAM(ClusterExtractor):
    """
    Explanator implementing CREAM algorithm.
    """

    def __init__(self, predictor, depth: int, dbscan_threshold: float, error_threshold: float,
                 constant: bool = False, seed=get_default_random_seed()):
        super().__init__(predictor)
        self.depth = depth
        self.dbscan_threshold = dbscan_threshold
        self.error_threshold = error_threshold
        self._constant = constant
        self.__generator = rnd.Random(seed)

    def extract(self, dataframe: pd.DataFrame) -> Theory:
        self._hypercubes = Node(dataframe, HyperCube.create_surrounding_cube(dataframe, True, self._constant))
        self._iterate(self._hypercubes)
        # return self._create_theory(dataframe)
        return None

    def _predict(self, data: dict[str, float]) -> float:
        data = {k: v for k, v in data.items()}
        return HyperCubeExtractor._get_cube_output(self._hypercubes.search(data), data)

    def _iterate(self, surrounding: Node) -> None:
        to_split = [(1, 1, surrounding)]
        while len(to_split) > 0:
            to_split.sort(reverse=True)
            (_, depth, node) = to_split.pop()
            gauss_pred = GaussianMixture(n_components=2).fit_predict(node.dataframe)

            cubes = []
            for i in range(2):
                df = node.dataframe.iloc[np.where(gauss_pred == i)]
                dbscan_pred = DBSCAN(eps=self.dbscan_threshold).fit_predict(df.iloc[:, :-1])
                cube = HyperCube.create_surrounding_cube(
                    df.iloc[np.where(dbscan_pred == Counter(dbscan_pred).most_common(1)[0][0])], True, self._constant
                )
                cubes.append((cube.volume(), i, cube))

            cube = min(cubes)[2]
            indices = cube.filter_indices(node.dataframe.iloc[:, :-1])
            if len(node.dataframe.iloc[indices]) * len(node.dataframe.iloc[~indices]) == 0:
                continue
            cube.update(node.dataframe.iloc[indices], self.predictor)
            node.right = Node(node.dataframe.iloc[indices], cube)
            node.cube.update(node.dataframe.iloc[~indices], self.predictor)
            node.left = Node(node.dataframe.iloc[~indices], node.cube)

            plt.scatter(node.dataframe.X, node.dataframe.Y,
                        c=gauss_pred, s=.5)
            plt.gca().set_aspect('equal')
            plt.xlim((0, 1))
            plt.ylim((0, 1))
            plt.show()
            for n in node.children:
                error = self._calculate_error(n.dataframe.iloc[:, :-1], n.cube)
                if error > self.error_threshold and depth < self.depth:
                    to_split.append((error, depth + 1, n))

    def _calculate_error(self, dataframe: pd.DataFrame, cube: ClosedCube) -> float:
        output = cube.output
        if isinstance(output, float):
            return np.mean(abs(self.predictor.predict(dataframe) - output))
        elif isinstance(output, LinearRegression):
            return np.mean(abs(self.predictor.predict(dataframe) - output.predict(dataframe)))

    @property
    def n_rules(self):
        return self._hypercubes.leaves
