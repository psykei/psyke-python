from __future__ import annotations

import math
from abc import ABC
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.linear_model import LinearRegression
from tuprolog.core import Var, Struct, clause
from tuprolog.theory import Theory, mutable_theory
from psyke.extraction import PedagogicalExtractor
from psyke.extraction.hypercubic.hypercube import HyperCube, RegressionCube, ClassificationCube, ClosedCube, Point, \
    GenericCube
from psyke.hypercubepredictor import HyperCubePredictor
from psyke.schema import Between, Outside, Value
from psyke.utils.logic import create_variable_list, create_head, to_var, Simplifier
from psyke.utils import Target
from psyke.extraction.hypercubic.strategy import Strategy, FixedStrategy


class HyperCubeExtractor(HyperCubePredictor, PedagogicalExtractor, ABC):
    def __init__(self, predictor, output, discretization=None, normalization=None):
        HyperCubePredictor.__init__(self, output=output, normalization=normalization)
        PedagogicalExtractor.__init__(self, predictor, discretization=discretization, normalization=normalization)
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

    def extract(self, dataframe: pd.DataFrame) -> Theory:
        theory = PedagogicalExtractor.extract(self, dataframe)
        self._surrounding = HyperCube.create_surrounding_cube(dataframe, output=self._output)
        self._surrounding.update(dataframe, self.predictor)
        return theory

    def pairwise_fairness(self, data: dict[str, float], neighbor: dict[str, float]):
        cube1 = self._find_cube(data)
        cube2 = self._find_cube(neighbor)
        different_prediction_reasons = []

        if cube1.output == cube2.output:
            print("Prediction", cube1.output, "is FAIR")
        else:
            print("Prediction", cube1.output, "may be UNFAIR")
            print("It could be", cube2.output, "if:")
            for d in data:
                a, b = cube2.dimensions[d]
                if data[d] < a:
                    print('    ', d, 'increases above', round(a, 1))
                    different_prediction_reasons.append(d)
                elif data[d] > b:
                    print('    ', d, 'decreases below', round(b, 1))
                    different_prediction_reasons.append(d)
        return different_prediction_reasons

    def predict_counter(self, data: dict[str, float], verbose=True, only_first=True):
        output = ""
        prediction = None
        cube = self._find_cube(data)
        if cube is None:
            output += "The extracted knowledge is not exhaustive; impossible to predict this instance"
        else:
            prediction = self._predict_from_cubes(data)
            output += f"The output is {prediction}\n"

        point = Point(list(data.keys()), list(data.values()))
        cubes = self._hypercubes if cube is None else [c for c in self._hypercubes if cube.output != c.output]
        cubes = sorted([(cube.surface_distance(point), cube.volume(), i, cube) for i, cube in enumerate(cubes)])

        counter_conditions = []

        for _, _, _, c in cubes:
            if not only_first or c.output not in [o for o, _ in counter_conditions]:
                counter_conditions.append((c.output, {c: [val for val in v if val is not None and not val.is_in(
                    self.unscale(data[c], c))] for c, v in self.__get_conditions(data, c).items()}))

        if verbose:
            for o, conditions in counter_conditions:
                output += f"The output may be {o} if\n" + HyperCubeExtractor.__conditions_to_string(conditions)
            print(output)

        return prediction, counter_conditions

    @staticmethod
    def __conditions_to_string(conditions: dict[str, list[Value]]) -> str:
        output = ""
        for d in conditions:
            for i, condition in enumerate(conditions[d]):
                if i == 0:
                    output += f'     {d} is '
                else:
                    output += ' and '
                output += condition.print()
                if i + 1 == len(conditions[d]):
                    output += '\n'
        return output

    def __get_conditions(self, data: dict[str, float], cube: GenericCube) -> dict[str, list[Value]]:
        conditions = {d: [cube.interval_to_value(d, self.unscale)] for d in data.keys()
                      if d not in self._dimensions_to_ignore}
        for c in cube.subcubes(self._hypercubes):
            for d in conditions:
                condition = c.interval_to_value(d, self.unscale)
                if condition is None:
                    continue
                elif conditions[d][-1] is None:
                    conditions[d][-1] = -condition
                else:
                    try:
                        conditions[d][-1] *= -condition
                    except Exception:
                        conditions[d].append(-condition)
        return conditions

    def predict_why(self, data: dict[str, float], verbose=True):
        cube = self._find_cube(data)
        output = ""
        if cube is None:
            output += "The extracted knowledge is not exhaustive; impossible to predict this instance\n"
            if verbose:
                print(output)
            return None, {}
        prediction = self._predict_from_cubes(data)
        output += f"The output is {prediction} because\n"
        conditions = {c: [val for val in v if val is not None and val.is_in(self.unscale(data[c], c))]
                      for c, v in self.__get_conditions(data, cube).items()}

        if verbose:
            output += HyperCubeExtractor.__conditions_to_string(conditions)
            print(output)

        return prediction, conditions

    @staticmethod
    def _create_head(dataframe: pd.DataFrame, variables: list[Var], output: float | LinearRegression) -> Struct:
        return create_head(dataframe.columns[-1], variables[:-1], output) \
            if not isinstance(output, LinearRegression) else \
            create_head(dataframe.columns[-1], variables[:-1], variables[-1])

    def __drop(self, dataframe: pd.DataFrame):
        self._hypercubes = [cube for cube in self._hypercubes if cube.count(dataframe) > 1]

    def _create_theory(self, dataframe: pd.DataFrame) -> Theory:
        # self.__drop(dataframe)
        for cube in self._hypercubes:
            for dimension in cube.dimensions:
                if abs(cube[dimension][0] - self._surrounding[dimension][0]) < HyperCube.EPSILON * 2:
                    cube.set_infinite(dimension, '-')
                if abs(cube[dimension][1] - self._surrounding[dimension][1]) < HyperCube.EPSILON * 2:
                    cube.set_infinite(dimension, '+')

        if self._default_surrounding_cube:
            self._hypercubes[-1].set_default()

        new_theory = mutable_theory()
        for cube in self._hypercubes:
            variables = create_variable_list([], dataframe)
            variables[dataframe.columns[-1]] = to_var(dataframe.columns[-1])
            head = HyperCubeExtractor._create_head(dataframe, list(variables.values()),
                                                   self.unscale(cube.output, dataframe.columns[-1]))
            body = cube.body(variables, self._dimensions_to_ignore, self.unscale, self.normalization)
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
