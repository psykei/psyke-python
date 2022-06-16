from __future__ import annotations
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from tuprolog.core import Var, Struct, clause
from tuprolog.theory import Theory, mutable_theory
from psyke import Extractor, logger
from psyke.regression.strategy import FixedStrategy, Strategy
from psyke.regression.utils import Limit, MinUpdate, ZippedDimension, Expansion
from psyke.schema import Between
from psyke.utils.logic import create_term, create_variable_list, create_head, to_var
from psyke.regression.hypercube import HyperCube, ClosedCube, RegressionCube, ClosedRegressionCube


class HyperCubeExtractor(Extractor):

    def __init__(self, predictor):
        super().__init__(predictor)
        self._hypercubes = []

    def extract(self, dataframe: pd.DataFrame) -> Theory:
        raise NotImplementedError('extract')

    def predict(self, dataframe: pd.DataFrame) -> Iterable:
        return np.array([self._predict(dict(row.to_dict())) for _, row in dataframe.iterrows()])

    def _predict(self, data: dict[str, float]) -> float:
        data = {k: v for k, v in data.items()}
        for cube in self._hypercubes:
            if cube.__contains__(data):
                return HyperCubeExtractor._get_cube_output(cube, data)
        return np.nan

    def _default_cube(self) -> HyperCube:
        return HyperCube()

    @staticmethod
    def _get_cube_output(cube: HyperCube | RegressionCube, data: dict[str, float]) -> float:
        return cube.output.predict(pd.DataFrame([data])).flatten()[0] if \
            isinstance(cube, RegressionCube) else cube.output

    @staticmethod
    def __create_body(variables: dict[str, Var], dimensions: dict[str, (float, float)]) -> Iterable[Struct]:
        return [create_term(variables[name], Between(values[0], values[1])) for name, values in dimensions.items()]

    @staticmethod
    def __create_head(dataframe: pd.DataFrame, variables: list[Var], output: float | LinearRegression) -> Struct:
        return create_head(dataframe.columns[-1], variables[:-1], output) \
            if not isinstance(output, LinearRegression) else \
            create_head(dataframe.columns[-1], variables[:-1], variables[-1])

    def _create_output(self, variables, target) -> None:
        return None

    def _create_theory(self, dataframe: pd.DataFrame) -> Theory:
        new_theory = mutable_theory()
        for cube in self._hypercubes:
            logger.info(cube.output)
            logger.info(cube.dimensions)
            variables = create_variable_list([], dataframe)
            variables[dataframe.columns[-1]] = to_var(dataframe.columns[-1])
            head = HyperCubeExtractor.__create_head(dataframe, list(variables.values()), cube.output)
            body = HyperCubeExtractor.__create_body(variables, cube.dimensions)
            output = self._create_output(list(variables.values()), cube.output)
            if output is not None:
                body += [output]
            new_theory.assertZ(
                clause(
                    head,
                    body
                )
            )
        return new_theory

    @property
    def n_rules(self):
        return len(self._hypercubes)


class FeatureRanker:
    def __init__(self, feat):
        self.scores = None
        self.feat = feat

    def fit(self, model, samples):
        best = SelectKBest(score_func=f_regression, k="all").fit(samples, model.predict(samples).flatten())
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
        if self.right.cube.__contains__(point):
            return self.right.search(point)
        return self.left.search(point)

    @property
    def leaves(self):
        if self.right is None:
            return 1
        return self.right.leaves + self.left.leaves
