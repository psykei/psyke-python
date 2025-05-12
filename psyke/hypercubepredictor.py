from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from psyke import EvaluableModel, Target, get_int_precision
from psyke.extraction.hypercubic import RegressionCube, GenericCube, Point


class HyperCubePredictor(EvaluableModel):
    def __init__(self, output=Target.CONSTANT, discretization=None, normalization=None):
        super().__init__(discretization, normalization)
        self._hypercubes = []
        self._dimensions_to_ignore = set()
        self._output = output
        self._surrounding = None

    def _predict(self, dataframe: pd.DataFrame) -> Iterable:
        return np.array([self._predict_from_cubes(row.to_dict()) for _, row in dataframe.iterrows()])

    def _brute_predict(self, dataframe: pd.DataFrame, criterion: str = 'corner', n: int = 2) -> Iterable:
        predictions = np.array(self._predict(dataframe))
        idx = [prediction is None for prediction in predictions]
        if sum(idx) > 0:
            if criterion == 'default':
                predictions[idx] = np.array([HyperCubePredictor._get_cube_output(
                    self._surrounding, row
                ) for _, row in dataframe[idx].iterrows()])
            elif criterion == 'surface':
                predictions[idx] = np.array([HyperCubePredictor._get_cube_output(self._brute_predict_surface(row), row)
                                             for _, row in dataframe[idx].iterrows()])
            else:
                tree, cubes = self._create_brute_tree(criterion, n)
                predictions[idx] = np.array([HyperCubePredictor._brute_predict_from_cubes(
                    row.to_dict(), tree, cubes
                ) for _, row in dataframe[idx].iterrows()])
        return np.array(predictions)

    @staticmethod
    def _brute_predict_from_cubes(row: dict[str, float], tree: BallTree,
                                  cubes: list[GenericCube]) -> float | str:
        idx = tree.query([list(row.values())], k=1)[1][0][0]
        return HyperCubePredictor._get_cube_output(cubes[idx], row)

    def _brute_predict_surface(self, row: pd.Series) -> GenericCube:
        return min([(
            cube.surface_distance(Point(list(row.keys()), list(row.values))), cube.volume(), cube
        ) for cube in self._hypercubes])[-1]

    def _create_brute_tree(self, criterion: str = 'center', n: int = 2) -> (BallTree, list[GenericCube]):
        admissible_criteria = ['surface', 'center', 'corner', 'perimeter', 'density', 'default']
        if criterion not in admissible_criteria:
            raise NotImplementedError(
                "'criterion' should be chosen in " + str(admissible_criteria)
            )

        points = [(cube.center, cube) for cube in self._hypercubes] if criterion == 'center' else \
            [(cube.barycenter, cube) for cube in self._hypercubes] if criterion == 'density' else \
            [(corner, cube) for cube in self._hypercubes for corner in cube.corners()] if criterion == 'corner' else \
            [(point, cube) for cube in self._hypercubes for point in cube.perimeter_samples(n)] \
            if criterion == 'perimeter' else None

        return BallTree(pd.concat([point[0].to_dataframe() for point in points], ignore_index=True)), \
            [point[1] for point in points]

    def _predict_from_cubes(self, data: dict[str, float]) -> float | str | None:
        cube = self._find_cube(data)
        if cube is None:
            return None
        elif self._output == Target.CLASSIFICATION:
            return HyperCubePredictor._get_cube_output(cube, data)
        else:
            return round(HyperCubePredictor._get_cube_output(cube, data), get_int_precision())

    def _find_cube(self, data: dict[str, float]) -> GenericCube | None:
        if not self._hypercubes:
            return None
        data = data.copy()
        for dimension in self._dimensions_to_ignore:
            if dimension in data:
                del data[dimension]
        for cube in self._hypercubes:
            if data in cube:
                return cube.copy()
        if self._hypercubes[-1].is_default:
            return self._hypercubes[-1].copy()

    @property
    def n_rules(self):
        return len(list(self._hypercubes))

    @property
    def volume(self):
        return sum([cube.volume() for cube in self._hypercubes])

    @staticmethod
    def _get_cube_output(cube, data: dict[str, float]) -> float:
        return cube.output.predict(pd.DataFrame([data])).flatten()[0] if \
            isinstance(cube, RegressionCube) else cube.output
