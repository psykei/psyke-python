from __future__ import annotations
import numpy as np
from psyke.regression import Node
from psyke.regression.creepy import CReEPy
from psyke.regression.utils import select_gaussian_mixture


class CREAM(CReEPy):
    """
    Explanator implementing CREAM algorithm.
    """

    def __init__(self, predictor, depth: int, error_threshold: float,
                 gauss_components: int = 5, constant: bool = False):
        super().__init__(predictor, depth, error_threshold, gauss_components, constant)

    def __eligible_cubes(self, gauss_pred: np.ndarray, node: Node, clusters: int):
        cubes = []
        for i in range(len(np.unique(gauss_pred))):
            df = node.dataframe.iloc[np.where(gauss_pred == i)]
            if len(df) == 0:
                continue
            inner_cube = self._create_cube(df, clusters)
            indices = self._indices(inner_cube, node.dataframe)
            if indices is None:
                continue
            right, left = self._split(inner_cube, node.cube, node.dataframe, indices)
            cubes.append((
                ((right.diversity + left.diversity) / 2, right.volume(), left.volume(), i),
                (right, indices), (left, ~indices)
            ))
        return cubes

    def _iterate(self, surrounding: Node) -> None:
        to_split = [(self.error_threshold * 10, 1, 1, surrounding)]
        while len(to_split) > 0:
            to_split.sort(reverse=True)
            (_, depth, _, node) = to_split.pop()
            gauss_params = select_gaussian_mixture(node.dataframe, self.gauss_components)
            gauss_pred = gauss_params[2].predict(node.dataframe)
            cubes = self.__eligible_cubes(gauss_pred, node, gauss_params[1])
            if len(cubes) < 1:
                continue
            _, right, left = min(cubes)
            node.right = Node(node.dataframe[right[1]], right[0])
            node.cube.update(node.dataframe[left[1]], self.predictor)
            node.left = Node(node.dataframe[left[1]], left[0])

            if depth < self.depth:
                to_split += [
                    (error, depth + 1, np.random.uniform(), n) for (n, error) in
                    zip(node.children, [right[0].diversity, left[0].diversity]) if error > self.error_threshold
                ]
