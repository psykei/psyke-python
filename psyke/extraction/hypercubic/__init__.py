from __future__ import annotations

from abc import ABC
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.linear_model import LinearRegression
from tuprolog.core import Var, Struct, clause
from tuprolog.theory import Theory, mutable_theory
from psyke import logger, PedagogicalExtractor
from psyke.extraction.hypercubic.hypercube import HyperCube, RegressionCube, ClassificationCube, ClosedCube, Point, \
    GenericCube
from psyke.utils.logic import create_variable_list, create_head, to_var, Simplifier
from psyke.utils import Target, get_int_precision
from psyke.extraction.hypercubic.strategy import Strategy, FixedStrategy
from sklearn.neighbors import BallTree


class HyperCubePredictor:
    def __init__(self, output=Target.CONSTANT, normalization=None):
        self._hypercubes = []
        self._output = output
        self.normalization = normalization

    def _predict(self, dataframe: pd.DataFrame) -> Iterable:
        return np.array([self._predict_from_cubes(row.to_dict()) for _, row in dataframe.iterrows()])

    def brute_predict(self, dataframe: pd.DataFrame, criterion: str = 'corner', n: int = 2) -> Iterable:
        predictions = self._predict(dataframe)
        idx = [prediction is None for prediction in predictions]

        tree, mapping = self._create_brute_tree(dataframe, criterion, n)

        predictions[idx] = np.array([HyperCubePredictor._brute_predict_from_cubes(
            row.to_dict(), tree, mapping
        ) for _, row in dataframe[idx].iterrows()])
        return predictions

    @staticmethod
    def _brute_predict_from_cubes(row: dict[str, float], tree: BallTree,
                                  mapping: dict[int, GenericCube]) -> float | str:
        idx = tree.query([list(row.values())], k=1)[1][0][0]
        return HyperCubePredictor._get_cube_output(mapping[idx], row)

    def _create_brute_tree(self, dataframe: pd.DataFrame,
                           criterion: str = 'center', n: int = 2) -> (BallTree, dict[int, GenericCube]):
        points = None
        if criterion == 'center':
            points = [(cube.center(), i, cube) for i, cube in enumerate(self._hypercubes)]
        elif criterion == 'density':
            points = [(cube.filter_dataframe(dataframe).describe().loc['mean'], i, cube)
                      for i, cube in enumerate(self._hypercubes)]
            points = [(Point(df.index.values, df.values), i, cube) for df, i, cube in points]
        elif criterion == 'corner':
            points = [(corner, cube) for cube in self._hypercubes for corner in cube.corners()]
            points = [(point[0], i, point[1]) for i, point in enumerate(points)]
        elif criterion == 'perimeter':
            points = [(point, cube) for cube in self._hypercubes for point in cube.perimeter_samples(n)]
            points = [(point[0], i, point[1]) for i, point in enumerate(points)]
        else:
            raise NotImplementedError("'criterion' should be chosen in ['center', 'corner', 'perimeter', 'density']")

        return BallTree(pd.concat([point[0].to_dataframe() for point in points], ignore_index=True), leaf_size=2), \
            {point[1]: point[2] for point in points}

    def _predict_from_cubes(self, data: dict[str, float]) -> float | str | None:
        for cube in self._hypercubes:
            if data in cube:
                if self._output == Target.CLASSIFICATION:
                    return HyperCubePredictor._get_cube_output(cube, data)
                else:
                    return round(HyperCubePredictor._get_cube_output(cube, data), get_int_precision())
        return None

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


class HyperCubeExtractor(HyperCubePredictor, PedagogicalExtractor, ABC):
    def __init__(self, predictor, output, discretization=None, normalization=None):
        PedagogicalExtractor.__init__(self, predictor, discretization=discretization, normalization=normalization)
        HyperCubePredictor.__init__(self, output=output, normalization=normalization)

    def _default_cube(self) -> HyperCube | RegressionCube | ClassificationCube:
        if self._output == Target.CONSTANT:
            return HyperCube()
        if self._output == Target.REGRESSION:
            return RegressionCube()
        return ClassificationCube()

    @staticmethod
    def _create_head(dataframe: pd.DataFrame, variables: list[Var], output: float | LinearRegression) -> Struct:
        return create_head(dataframe.columns[-1], variables[:-1], output) \
            if not isinstance(output, LinearRegression) else \
            create_head(dataframe.columns[-1], variables[:-1], variables[-1])

    def _ignore_dimensions(self) -> Iterable[str]:
        return []

    def _create_theory(self, dataframe: pd.DataFrame, sort: bool = True) -> Theory:
        new_theory = mutable_theory()
        for cube in self._hypercubes:
            logger.info(cube.output)
            logger.info(cube.dimensions)
            variables = create_variable_list([], dataframe, sort)
            variables[dataframe.columns[-1]] = to_var(dataframe.columns[-1])
            head = HyperCubeExtractor._create_head(dataframe, list(variables.values()),
                                                   self.unscale(cube.output, dataframe.columns[-1]))
            body = cube.body(variables, self._ignore_dimensions(), self.unscale, self.normalization)
            new_theory.assertZ(clause(head, body))
        return HyperCubeExtractor._prettify_theory(new_theory)

    @staticmethod
    def _prettify_theory(theory: Theory) -> Theory:
        visitor = Simplifier()
        new_clauses = []
        for c in theory.clauses:
            body = c.body
            structs = body.unfolded if c.body_size > 1 else [body]
            new_structs = []
            for s in structs:
                new_structs.append(s.accept(visitor))
            new_clauses.append(clause(c.head, new_structs))
        return mutable_theory(new_clauses)


class FeatureRanker:
    def __init__(self, feat):
        self.scores = None
        self.feat = feat

    def fit(self, model, samples):
        predictions = np.array(model.predict(samples)).flatten()
        function = f_classif if isinstance(model, ClassifierMixin) else f_regression
        best = SelectKBest(score_func=function, k="all").fit(samples, predictions)
        self.scores = np.array(best.scores_) / max(best.scores_)
        return self

    def fit_on_data(self, samples):
        function = f_classif if isinstance(samples.iloc[0, -1], str) else f_regression
        best = SelectKBest(score_func=function, k="all").fit(samples.iloc[:, :-1], samples.iloc[:, -1])
        self.scores = np.array(best.scores_) / max(best.scores_)
        return self

    def rankings(self):
        return list(zip(self.feat, self.scores))


class Grid:
    def __init__(self, iterations: int = 1, strategy: Strategy | list[Strategy] = FixedStrategy()):
        self.iterations = iterations
        self.strategy = strategy

    def get(self, feature: str, depth: int) -> int:
        if isinstance(self.strategy, list):
            return self.strategy[depth].get(feature)
        else:
            return self.strategy.get(feature)

    def iterate(self) -> range:
        return range(self.iterations)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Grid ({}). {}".format(self.iterations, self.strategy)


class Node:
    def __init__(self, dataframe: pd.DataFrame, cube: ClosedCube = None):
        self.dataframe = dataframe
        self.cube: ClosedCube = cube
        self.right: Node | None = None
        self.left: Node | None = None

    @property
    def children(self) -> list[Node]:
        return [self.right, self.left]

    def search(self, point: dict[str, float]) -> ClosedCube:
        if self.right is None:
            return self.cube
        if point in self.right.cube:
            return self.right.search(point)
        return self.left.search(point)

    @property
    def leaves(self):
        if self.right is None:
            return 1
        return self.right.leaves + self.left.leaves
