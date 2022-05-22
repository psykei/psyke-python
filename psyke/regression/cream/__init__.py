from __future__ import annotations
import numpy as np
from numpy import ndarray
from sklearn.mixture import GaussianMixture
from tuprolog.theory import Theory
from psyke.regression import Node
from psyke.regression.creepy import CReEPy


class CREAM(CReEPy):
    """
    Explanator implementing CREAM algorithm.
    """

    def __init__(self, predictor, depth: int, dbscan_threshold: float,
                 error_threshold: float, gauss_components: int = 2, constant: bool = False):
        super().__init__(predictor, depth, dbscan_threshold, error_threshold, gauss_components, constant)

    def __eligible_cubes(self, gauss_pred: ndarray, node: Node):
        cubes = []
        for inner_cube in [
            self._create_cube(node.dataframe.iloc[np.where(gauss_pred == i)]) for i in range(len(np.unique(gauss_pred)))
        ]:
            indices = self._indices(inner_cube, node.dataframe)
            if indices is None:
                continue
            right, left = self._split(inner_cube, node.cube, node.dataframe, indices)
            cubes.append((
                ((right.diversity + left.diversity) / 2, right.volume(), left.volume()),
                (right, indices), (left, ~indices)
            ))
        return cubes

    def _iterate(self, surrounding: Node) -> None:
        to_split = [(self.error_threshold * 10, 1, surrounding)]
        while len(to_split) > 0:
            to_split.sort(reverse=True)
            (_, depth, node) = to_split.pop()
            components = max(self.gauss_components - depth + 1, 2)
            gauss_pred = GaussianMixture(n_components=components).fit_predict(node.dataframe)
            cubes = self.__eligible_cubes(gauss_pred, node)
            if len(cubes) < 1:
                continue
            _, right, left = min(cubes)
            node.right = Node(node.dataframe[right[1]], right[0])
            node.cube.update(node.dataframe[left[1]], self.predictor)
            node.left = Node(node.dataframe[left[1]], left[0])

            if depth < self.depth:
                to_split += [
                    (error, depth + 1, n) for (n, error) in
                    zip(node.children, [right[0].diversity, left[0].diversity]) if error > self.error_threshold
                ]
