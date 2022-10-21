from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd

from psyke.clustering.exact import ExACT
from psyke.clustering.orbit.container import Container, ContainerNode
from psyke.clustering.orbit.oblique_rules_generator import generate_container
from psyke.extraction.hypercubic import Node, ClosedCube, HyperCube
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import DBSCAN
from psyke.clustering.utils import select_gaussian_mixture, select_dbscan_epsilon
from collections import Counter
from psyke.utils import Target
from sklearn.metrics import accuracy_score


class MixedRulesExtractor:
    """extractor for mixed rules: hyper-cubes and oblique rules"""

    def __init__(self, depth: int, error_threshold: float, gauss_components: int = 5, steps=1000,
                 min_accuracy_increase=0.01, max_disequation_num=4):
        """

        :param depth: depth of the tree of rules (contraints) that will be generated
        :param error_threshold:
        :param gauss_components: number of gaussian clusters used to split data into different hyper-cubes/oblique rules
        :param steps: every time disequations are created,
            only steps couples of dimensions are checked to generate disequations
        :param min_accuracy_increase: oblique rules (diequtions) are preferred to hypercubes only if cause an increse
            in accuracy of "min_accuracy_increase"
        """
        self.depth = depth
        self.error_threshold = error_threshold
        self.gauss_components = gauss_components
        self._containers = None
        # super(CREAM2, self).__init__(depth, error_threshold, output, gauss_components)
        self._predictor = KNeighborsClassifier()
        self._predictor.n_neighbors = 1
        self._output = Target.CLASSIFICATION
        self.steps = steps
        self.min_accuracy_increase = min_accuracy_increase
        self.inistial_dataset_size = 0
        self.max_disequation_num = 4

    def extract(self, dataframe: pd.DataFrame) -> List[Container]:
        self._predictor.fit(dataframe.iloc[:, :-1], dataframe.iloc[:, -1])
        self._containers = \
            self._iterate(ContainerNode(dataframe, Container.create_surrounding_cube(dataframe, True)))
        return list(self._containers)

    def _iterate(self, surrounding: ContainerNode) -> List[HyperCube]:
        self.inistial_dataset_size = len(surrounding.dataframe.index)
        to_split = [(self.error_threshold * 10, 1, 1, surrounding, surrounding.dataframe)]
        while len(to_split) > 0:
            to_split.sort(reverse=True)
            (_, depth, _, node, domain_data) = to_split.pop()
            assert isinstance(node, ContainerNode)
            data = ExACT._remove_string_label(node.dataframe)
            data.iloc[:, -1] = data.iloc[:, -1]
            gauss_params = select_gaussian_mixture(data, self.gauss_components)
            gauss_pred = gauss_params[2].predict(data)
            # from utils.draw import draw_data, draw_clusters
            # draw_clusters(node.dataframe.iloc[:, :-1], node.dataframe.iloc[:, -1])
            containers = self.__eligible_cubes(domain_data, gauss_pred, node, gauss_params[1])
            if len(containers) < 1:
                continue
            _, (right_container, right_indices), (left_container, left_indices) = min(containers, key=lambda x: x[0][0])

            node.right = ContainerNode(node.dataframe[right_indices], right_container)
            node.container.update(node.dataframe[left_indices], self._predictor)
            node.left = ContainerNode(node.dataframe[left_indices], left_container)
            # data = node.right.container._filter_dataframe(node.right.dataframe)
            # draw_clusters(data.iloc[:, :-1], data.iloc[:, -1])
            # data = node.left.container._filter_dataframe(node.left.dataframe)
            # draw_clusters(data.iloc[:, :-1], data.iloc[:, -1])


            if depth < self.depth:
                to_split += [
                    (error, depth + 1, np.random.uniform(), n, d_d) for (n, error, d_d) in
                    zip(node.children, [right_container.diversity, left_container.diversity], [domain_data, node.left.dataframe]) if error > self.error_threshold
                ]
        return self._node_to_cubes(surrounding)

    def __eligible_cubes(self, dataframe: pd.DataFrame, gauss_pred: np.ndarray, node: ContainerNode, clusters: int):
        prediction_value = node.container.output
        df_without_clusters = dataframe.iloc[:, :-1]
        real_pred = (dataframe.iloc[:, -1] == prediction_value).to_numpy()
        # predictions of parent node
        parent_node_pred = node.container.filter_indices(df_without_clusters)
        parent_accuracy = accuracy_score(real_pred, parent_node_pred)

        cubes = []
        for i in range(len(np.unique(gauss_pred))):
            for c in node.dataframe[gauss_pred == i].iloc[:, -1].unique():
                df = node.dataframe[np.logical_and(gauss_pred == i, node.dataframe.iloc[:, -1] == c)]
                if len(df) == 0:
                    continue
                inner_cube = self._create_cube(df, clusters)
                cube_indices = self._indices(inner_cube, node.dataframe)
                if cube_indices is None:
                    continue
                inner_container = generate_container(dataframe,
                                                     node.dataframe,
                                                     cube_indices,
                                                     inner_cube,
                                                     steps=self.steps,
                                                     min_accuracy_increase=self.min_accuracy_increase,
                                                     initial_size=self.inistial_dataset_size,
                                                     max_disequation_num=self.max_disequation_num
                                                     )
                indices = self._indices(inner_container, node.dataframe)
                if indices is None:
                    continue
                right, left = self._split(inner_container, node.cube, node.dataframe, indices)
                # if output of left and right parts are the same
                exclude_cubes = False
                if left.output == right.output:
                    # check if left and right cube will increase performances
                    left_pred = left.filter_indices(df_without_clusters)
                    right_pred = right.filter_indices(df_without_clusters)

                    # combined predictions of left and right cubes
                    left_right_pred = np.logical_or(left_pred, right_pred)

                    children_accuracy = accuracy_score(real_pred, left_right_pred)
                    if children_accuracy <= parent_accuracy:
                        exclude_cubes = True
                if not exclude_cubes:
                    # cubes.append((
                    #     (max(right.diversity, left.diversity) / 2, right.volume(), left.volume(), i),
                    #     (right, indices), (left, ~indices)
                    # ))
                    cubes.append((
                        ((right.diversity + left.diversity) / 2, right.volume(), left.volume(), i),
                        (right, indices), (left, ~indices)
                    ))

        return cubes

    def _split(self, right: Container, outer_container: Container, data: pd.DataFrame, indices: np.ndarray):
        right.update(data.iloc[indices], self._predictor)
        left = outer_container.copy()
        left.update(data.iloc[~indices], self._predictor)
        return right, left

    def predict(self, dataframe: pd.DataFrame) -> Iterable:
        return np.array([self._predict(dict(row.to_dict())) for _, row in dataframe.iterrows()])

    def _predict(self, data: dict[str, float]) -> float | None:
        data = {k: v for k, v in data.items()}
        for container in self._containers:
            if container.__contains__(data):
                self._get_cube_output(container)
        return None

    def _get_cube_output(self, cube: Container) -> float:
        return cube.output

    def _node_to_cubes(self, root: ContainerNode) -> list[Container]:
        if root.right is None:
            return [root.container]
        else:
            return self._node_to_cubes(root.right) + self._node_to_cubes(root.left)

    def _create_cube(self, dataframe: pd.DataFrame, clusters: int) -> ClosedCube:
        data = ExACT._remove_string_label(dataframe)
        dbscan_pred = DBSCAN(eps=select_dbscan_epsilon(data, clusters)).fit_predict(data.iloc[:, :-1])
        return HyperCube.create_surrounding_cube(
            dataframe.iloc[np.where(dbscan_pred == Counter(dbscan_pred).most_common(1)[0][0])],
            True, self._output
        )

    @staticmethod
    def _indices(container: Container, data: pd.DataFrame) -> np.ndarray | None:
        indices = container.filter_indices(data.iloc[:, :-1])
        if len(data.iloc[indices]) * len(data.iloc[~indices]) == 0:
            return None
        return indices