from __future__ import annotations

from abc import ABC
from itertools import groupby
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.linear_model import LinearRegression
from tuprolog.core import Var, Struct, clause
from tuprolog.theory import Theory, mutable_theory
from psyke import logger
from psyke.extraction import PedagogicalExtractor
from psyke.extraction.hypercubic.hypercube import HyperCube, RegressionCube, ClassificationCube, ClosedCube, Point, \
    GenericCube
from psyke.hypercubepredictor import HyperCubePredictor
from psyke.schema import Between, Outside, Value
from psyke.utils.logic import create_variable_list, create_head, to_var, Simplifier, last_in_body, PRECISION
from psyke.utils import Target
from psyke.extraction.hypercubic.strategy import Strategy, FixedStrategy


class HyperCubeExtractor(HyperCubePredictor, PedagogicalExtractor, ABC):
    def __init__(self, predictor, output, discretization=None, normalization=None):
        HyperCubePredictor.__init__(self, output=output, normalization=normalization)
        PedagogicalExtractor.__init__(self, predictor, discretization=discretization, normalization=normalization)
        self._surrounding = None
        self._default_surrounding_cube = False

    def _default_cube(self) -> HyperCube | RegressionCube | ClassificationCube:
        if self._output == Target.CONSTANT:
            return HyperCube()
        if self._output == Target.REGRESSION:
            return RegressionCube()
        return ClassificationCube()

    def _sort_cubes(self):
        cubes = [(cube.diversity, i, cube) for i, cube in enumerate(self._hypercubes)]
        cubes.sort()
        self._hypercubes = [cube[2] for cube in cubes]

    def _last_cube_as_default(self, theory):
        last_clause = list(theory.clauses)[-1]
        theory.retract(last_clause)
        theory.assertZ(clause(
            last_clause.head, [last_in_body(last_clause.body)] if self._output is Target.REGRESSION else []))
        last_cube = self._hypercubes[-1]
        for dimension in last_cube.dimensions.keys():
            last_cube[dimension] = [-np.inf, np.inf]
        return theory

    def extract(self, dataframe: pd.DataFrame) -> Theory:
        theory = PedagogicalExtractor.extract(self, dataframe)
        self._surrounding = HyperCube.create_surrounding_cube(dataframe, output=self._output)
        self._surrounding.update(dataframe, self.predictor)
        return theory

    def predict_counter(self, data: dict[str, float]):
        cube = self._find_cube(data)
        if cube is None:
            print("The extracted knowledge is not exhaustive; impossible to predict this instance")
        else:
            print("The output is", self._predict_from_cubes(data))

        point = Point(list(data.keys()), list(data.values()))
        cubes = self._hypercubes if cube is None else [c for c in self._hypercubes if cube.output != c.output]
        cubes = sorted([(cube.surface_distance(point), cube.volume(), cube) for cube in cubes])
        outputs = []
        for _, _, c in cubes:
            if c.output not in outputs:
                outputs.append(c.output)
                print("The output may be", c.output, 'if')

                for d in c.dimensions.keys():
                    lower, upper = c[d]
                    p = point[d]
                    if p < lower:
                        print('    ', d, '=', round(lower, 1))
                    elif p > upper:
                        print('    ', d, '=', round(upper, 1))

    def __get_local_conditions(self, cube: GenericCube) -> dict[list[Value]]:
        conditions = {d: [] for d in cube.dimensions}
        for d in cube.finite_dimensions:
            conditions[d].append(Between(*cube.dimensions[d]))
        subcubes = cube.subcubes(self._hypercubes)
        for c in [c for c in subcubes if sum(c in sc and c != sc for sc in subcubes) == 0]:
            for d in c.finite_dimensions:
                conditions[d].append(Outside(*c.dimensions[d]))
        return conditions

    def predict_why(self, data: dict[str, float]):
        cube = self._find_cube(data)
        if cube is None:
            print("The extracted knowledge is not exhaustive; impossible to predict this instance")
        else:
            output = self._predict_from_cubes(data)
            print(f"The output is {output} because")
            conditions = self.__get_local_conditions(cube)
            for d in conditions:
                simplified = HyperCubeExtractor.__simplify(conditions[d])
                for i, condition in enumerate(simplified):
                    if i == 0:
                        print('    ', d, 'is', end=' ')
                    else:
                        print('and', end=' ')
                    if isinstance(condition, Outside):
                        print('not', end=' ')
                    print('between', round(condition.lower, 1), 'and', round(condition.upper, 1), end=' ')
                    if i + 1 == len(simplified):
                        print()

    @staticmethod
    def __simplify(conditions):
        simplified = []
        for condition in conditions:
            to_add = True
            for i, simple in enumerate(simplified):
                if isinstance(condition, Outside) and isinstance(simple, Outside):
                    if simple.lower <= condition.lower <= simple.upper or \
                            simple.lower <= condition.upper <= simple.upper or \
                            condition.lower <= simple.lower <= simple.upper <= condition.upper:
                        simplified[i].upper = max(condition.upper, simple.upper)
                        simplified[i].lower = min(condition.lower, simple.lower)
                        to_add = False
                        break
                elif isinstance(condition, Outside) and isinstance(simple, Between):
                    if simple.lower >= condition.upper or simple.upper <= condition.lower:
                        to_add = False
                        break
                    elif condition.lower <= simple.lower <= condition.upper <= simple.upper:
                        simplified[i].lower = condition.upper
                        to_add = False
                        break
                    elif simple.lower <= condition.lower <= simple.upper <= condition.upper:
                        simplified[i].upper = condition.lower
                        to_add = False
                        break
                    elif condition.lower <= simple.lower <= simple.upper <= condition.upper:
                        raise ValueError
            if to_add:
                simplified.append(condition)
        return simplified

    @staticmethod
    def _create_head(dataframe: pd.DataFrame, variables: list[Var], output: float | LinearRegression) -> Struct:
        return create_head(dataframe.columns[-1], variables[:-1], output) \
            if not isinstance(output, LinearRegression) else \
            create_head(dataframe.columns[-1], variables[:-1], variables[-1])

    def _ignore_dimensions(self, cube: HyperCube) -> Iterable[str]:
        return [d for d in cube.dimensions if cube[d][0] == -np.inf or cube[d][1] == np.inf]

    def __drop(self, dataframe: pd.DataFrame):
        self._hypercubes = [cube for cube in self._hypercubes if cube.count(dataframe) > 1]

    def _create_theory(self, dataframe: pd.DataFrame) -> Theory:
        self.__drop(dataframe)
        new_theory = mutable_theory()
        for cube in self._hypercubes:
            logger.info(cube.output)
            logger.info(cube.dimensions)
            variables = create_variable_list([], dataframe)
            variables[dataframe.columns[-1]] = to_var(dataframe.columns[-1])
            head = HyperCubeExtractor._create_head(dataframe, list(variables.values()),
                                                   self.unscale(cube.output, dataframe.columns[-1]))
            body = cube.body(variables, self._ignore_dimensions(cube), self.unscale, self.normalization)
            new_theory.assertZ(clause(head, body))
        new_theory = HyperCubeExtractor._prettify_theory(new_theory)
        return self._last_cube_as_default(new_theory) if self._default_surrounding_cube else new_theory

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
