from __future__ import annotations
from statistics import mode
from functools import reduce
from typing import Iterable, Union, Tuple
import pandas as pd
from numpy import ndarray

from psyke.extraction.hypercubic.utils import MinUpdate, Limit, Expansion
from psyke.extraction.hypercubic.hypercube import ClosedClassificationCube
from psyke.schema import Between, LessThan
from psyke.utils import get_default_precision, get_int_precision, Target, get_default_random_seed
from psyke.extraction.hypercubic import Node
from psyke.extraction.hypercubic.hypercube import HyperCube
from psyke.utils.logic import create_term
from sklearn.linear_model import LinearRegression
from tuprolog.core import Var, Struct
from random import Random
import numpy as np

# todo: eliminate the following part
from tuprolog.core import Var, Struct, Real, Term, Integer, Numeric, clause
from psyke.schema import Value, LessThan, GreaterThan, Between, Constant, term_to_value, Outside
from tuprolog.core import struct, real, atom, var, numeric, logic_list, Clause
PRECISION: int = get_int_precision()
def get_int_precision() -> int:
    from math import log10
    return -1 * int(log10(get_default_precision()))


Constraint = tuple[float, float, float]
Constraints = dict[tuple[str, str], tuple[float, float, float]]


class FeatureNotFoundException(Exception):
    def __init__(self, feature: str):
        super().__init__('Feature "' + feature + '" not found.')


class OutputNotDefinedException(Exception):
    def __init__(self):
        super().__init__('The output of the container is not defined')


class ConstraintFeaturesNotFoundException(Exception):

    def __init__(self, feature: tuple[str, str]):
        super().__init__(f'Constraint feature {feature} not found.')


class ContainerNode(Node):
    def __init__(self, dataframe: pd.DataFrame, container: Container = None):
        super().__init__(dataframe, container)
        self.dataframe = dataframe
        self.container: Container = container
        self.right: ContainerNode | None = None
        self.left: ContainerNode | None = None
    @property
    def children(self) -> list[ContainerNode]:
        return [self.right, self.left]

    def search(self, point: dict[str, float]) -> Container:
        if self.right is None:
            return self.container
        if self.right.container.__contains__(point):
            return self.right.search(point)
        return self.left.search(point)

    @property
    def leaves(self):
        if self.right is None:
            return 1
        return self.right.leaves + self.left.leaves


class Container(ClosedClassificationCube):
    """
    A N-dimensional cube holding a numeric value.
    """

    EPSILON = get_default_precision()  # Precision used when comparing two hypercubes
    INT_PRECISION = get_int_precision()

    def __init__(self,
                 dimension: dict[str, tuple],
                 disequation: dict[tuple[str, str], list[tuple[float, float, float]]]={},
                 limits: set[Limit] = None,
                 convex_hulls: Tuple = ([], 0)):
        """

        :param disequation: is in the form (X,Y): a,b,c, which identifies the constraint aX + bY <= c,
            where X and Y are the names of the features that are being constrained
        :param limits:
        """
        self._disequations = disequation
        # self._constraints = self._fit_dimension(constraints) if constraints is not None else {}
        # self._limits = limits if limits is not None else set()
        # self._diversity = 0.0
        self._output = None
        self.convex_hulls = convex_hulls
        super().__init__(dimension=dimension)

    def update(self, dataset: pd.DataFrame, predictor) -> None:
        # filtered = self._filter_dataframe(dataset.iloc[:, :-1])
        filtered = self._filter_dataframe(dataset.iloc[:, :-1])
        if len(filtered > 0):
            predictions = predictor.predict(filtered)
            self._output = mode(predictions)
            self._diversity = 1 - sum(prediction == self.output for prediction in predictions) / len(filtered)

    def filter_indices(self, dataset: pd.DataFrame) -> ndarray:
        output = np.full(len(dataset.index), True, dtype=bool)
        for column in self.dimensions.keys():
            out = np.logical_and(dataset[column] >= self.dimensions[column][0], dataset[column] <= self.dimensions[column][1])
            if output is None:
                output = out
            else:
                output = np.logical_and(output, out)
        output = np.logical_and(output, Container.check_sat_constraints(self._disequations, dataset))
        if isinstance(output, pd.Series):
            output = output.to_numpy()
        return output

        # v = np.array([v for _, v in self._dimensions.items()])
        # ds = dataset.to_numpy(copy=True)
        # return np.all((v[:, 0] <= ds) & (ds < v[:, 1]), axis=1)

    def _filter_dataframe(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset[self.filter_indices(dataset)]

    @staticmethod
    def check_sat_constraints(constraints: dict[tuple[str, str], list[tuple[float, float, float]]], dataset) -> np.ndarray:
        output = np.full(len(dataset.index), True, dtype=bool)
        for constr_columns in constraints:
            col1 = constr_columns[0]
            col2 = constr_columns[1]
            for a, b, c in constraints[constr_columns]:
                out = dataset[col1] * a + dataset[col2] * b <= c
                if output is None:
                    output = out
                else:
                    output = output & out
        return output

    @staticmethod
    def create_surrounding_cube(dataset: pd.DataFrame, closed: bool = False,
                                output=None) -> Container:
        hyper_cube = HyperCube.create_surrounding_cube(dataset, closed)
        return Container(hyper_cube.dimensions)

    def copy(self) -> Container:
        return Container(self.dimensions.copy(), self._disequations.copy(), convex_hulls=self.convex_hulls)

    @property
    def diequations(self) -> dict[tuple[str, str], list[tuple[float, float, float]]]:
        return self._disequations

    def body(self, variables: dict[str, Var], ignore: list[str], unscale=None, normalization=None) -> Iterable[Struct]:
        """
        generate the body of the theory that describes this container
        :param variables:
        :param ignore:
        :param unscale:
        :param normalization:
        :return:
        """
        dimensions = dict(self.dimensions)
        constraints = self.diequations.copy()
        for dimension in ignore:
            del dimensions[dimension]
        for (dim1, dim2) in self.diequations:
            if dim1 in ignore or dim2 in ignore:
                del constraints[(dim1, dim2)]
        dimension_out = [create_term(variables[name], Between(unscale(values[0], name), unscale(values[1], name)))
                         for name, values in dimensions.items()]

        constr_out = []
        for dim1, dim2 in self.diequations:
            for a, b, c in self.diequations[(dim1, dim2)]:
                x = struct("*", real(round(a, PRECISION)), variables[dim1])
                y = struct("*", real(round(b, PRECISION)), variables[dim2])
                constr_out.append(struct("=<", struct("+", x, y), real(round(c, PRECISION))))

        return dimension_out + constr_out

    def __contains__(self, point: dict[str, float]) -> bool:
        """
        Note that a point (dict[str, float]) is inside a container if ALL its dimensions' values satisfy:
            aX + bY <= c
        :param point: an N-dimensional point
        :return: true if the point is inside the container, false otherwise
        """
        for X, Y in self.diequations.keys():
            x = point[X]
            y = point[Y]
            for a, b, c in self._disequations[(X, Y)]:
                if not(a * x + b * y <= c):
                    return False
        for dim in self.dimensions:
            start, end = self.dimensions[dim]
            if not( start <= point[dim] and point[dim] <= end):
                return False
        return True

    @property
    def output(self):
        if self._output is None:
            raise OutputNotDefinedException()
        else:
            return self._output

    # def __eq__(self, other: Container) -> bool:
    #
    #     for dims in self._constraints.keys():
    #         if dims not in other.constraints:
    #             return False
    #         else:
    #             other_a_b_c = list(other.constraints[dims])
    #             a_b_c = list(self._constraints[dims])
    #             if any([abs(param - other_param) > Container.EPSILON for param, other_param in zip(a_b_c, other_a_b_c)]):
    #                 return False
    #     return True
    #
    # def __getitem__(self, feature: tuple[str, str]) -> Constraint:
    #     if feature in self._constraints.keys():
    #         return self._constraints[feature]
    #     else:
    #         raise ConstraintFeaturesNotFoundException(feature)
    #
    # def __setitem__(self, key, value) -> None:
    #     self._constraints[key] = value
    #
    # def __hash__(self) -> int:
    #     result = [hash(name1 + name2 + str(constraint[0]) + str(constraint[1]) + str(constraint[2]))
    #               for (name1, name2), constraint in self.constraints.items()]
    #     return sum(result)

    def _fit_constraint(self, dimension: dict[tuple[str, str], tuple[float, float, float]]) -> dict[tuple[str, str], tuple[float, float, float]]:
        new_dimension: dict[tuple[str, str], tuple[float, float, float]] = {}
        for key, value in dimension.items():
            new_dimension[key] = (round(value[0], self.INT_PRECISION), round(value[1], self.INT_PRECISION), round(value[2], self.INT_PRECISION))
        return new_dimension

    # def _expand_one(self, update: MinUpdate, surrounding: Container, ratio: float = 1.0) -> None:
    #     # todo: it won't work. Is it needed?
    #     self.update_dimension(update.name, (
    #         max(self.get_first(update.name) - update.value / ratio, surrounding.get_first(update.name)),
    #         min(self.get_second(update.name) + update.value / ratio, surrounding.get_second(update.name))
    #     ))
    #
    # def filter_indices(self, dataset: pd.DataFrame) -> ndarray:
    #     # todo: it won't work. Is it needed?
    #     v = np.array([v for _, v in self._constraints.items()])
    #     ds = dataset.to_numpy(copy=True)
    #     return np.all((v[:, 0] <= ds) & (ds < v[:, 1]), axis=1)
    #
    # def _filter_dataframe(self, dataset: pd.DataFrame) -> pd.DataFrame:
    #     # todo: it won't work. Is it needed?
    #     return dataset[self.filter_indices(dataset)]
    #
    # def add_limit(self, limit_or_feature: Limit | str, direction: str = None) -> None:
    #     # todo: it won't work. Is it needed?
    #     if isinstance(limit_or_feature, Limit):
    #         self._limits.add(limit_or_feature)
    #     else:
    #         self.add_limit(Limit(limit_or_feature, direction))
    #
    # def check_limits(self, feature: str) -> str | None:
    #     # todo: it won't work. Is it needed?
    #     filtered = [limit for limit in self._limits if limit.feature == feature]
    #     if len(filtered) == 0:
    #         return None
    #     if len(filtered) == 1:
    #         return filtered[0].direction
    #     if len(filtered) == 2:
    #         return '*'
    #     raise Exception('Too many limits for this feature')
    #
    # def create_samples(self, n: int = 1, generator: Random = Random(get_default_random_seed())) -> pd.DataFrame:
    #     return pd.DataFrame([self._create_tuple(generator) for _ in range(n)])
    #
    # @staticmethod
    # def check_overlap(to_check: Iterable[Container], hypercubes: Iterable[Container]) -> bool:
    #     checked = []
    #     to_check_copy = list(to_check).copy()
    #     while len(to_check_copy) > 0:
    #         cube = to_check_copy.pop()
    #         for hypercube in hypercubes:
    #             if hypercube not in checked and cube.overlap(hypercube):
    #                 return True
    #         checked += [cube]
    #     return False
    #
    # def count(self, dataset: pd.DataFrame) -> int:
    #     # todo: it won't work. Is it needed?
    #     return self._filter_dataframe(dataset.iloc[:, :-1]).shape[0]
    #
    # def body(self, variables: dict[str, Var], ignore: list[tuple[str, str]], unscale=None, normalization=None) -> Iterable[Struct]:
    #     # todo: it won't work. Is it needed?
    #     constraints = dict(self.constraints)
    #     for constr in ignore:
    #         del constraints[constr]
    #     return [create_term(variables[name], Between(unscale(values[0], name), unscale(values[1], name)))
    #             for name, values in constraints.items()]
    #
    # @staticmethod
    # def create_surrounding_cube(dataset: pd.DataFrame, closed: bool = False) -> GenericContainer:
    #     raise Exception("need to implement")
    #     # constraints = {
    #     #     column: (min(dataset[column]) - Container.EPSILON * 2, max(dataset[column]) + Container.EPSILON * 2)
    #     #     for column in dataset.columns[:-1]
    #     # }
    #     # return Container(constraints)
    #
    #
    #     # if closed:
    #     #     if output == Target.CONSTANT:
    #     #         return Container(dimensions)
    #     #     if output == Target.REGRESSION:
    #     #         return ClosedRegressionCube(dimensions)
    #     #     return ClosedClassificationCube(dimensions)
    #     # if output == Target.CLASSIFICATION:
    #     #     return ClassificationCube(dimensions)
    #     # if output == Target.REGRESSION:
    #     #     return RegressionCube(dimensions)
    #     # return HyperCube(dimensions)
    #
    # def _create_tuple(self, generator: Random) -> dict:
    #     # todo: it won't work. Is it needed?
    #     return {k: generator.uniform(self.get_first(k), self.get_second(k)) for k in self._constraints.keys()}
    #
    # @staticmethod
    # def cube_from_point(point: dict, output=None) -> GenericContainer:
    #     # todo: it won't work. Is it needed?
    #     # if output is Target.CLASSIFICATION:
    #     #     return ClassificationCube({k: (v, v) for k, v in list(point.items())[:-1]})
    #     # if output is Target.REGRESSION:
    #     #     return RegressionCube({k: (v, v) for k, v in list(point.items())[:-1]})
    #     return Container({k: (v, v) for k, v in list(point.items())[:-1]}, output=list(point.values())[-1])
    #
    # def equal(self, containers: Iterable[Container] | Container) -> bool:
    #     if isinstance(containers, Iterable):
    #         return any([self.__eq__(cont) for cont in containers])
    #     else:
    #         return self.__eq__(containers)
    #
    # def expand(self, expansion: Expansion, hypercubes: Iterable[Container]) -> None:
    #     # todo: it won't work. Is it needed?
    #     feature = expansion.feature
    #     a, b = self[feature]
    #     self.update_dimension(feature, expansion.boundaries(a, b))
    #     other_cube = self.overlap(hypercubes)
    #     if isinstance(other_cube, Container):
    #         self.update_dimension(feature, (other_cube.get_second(feature), b)
    #         if expansion.direction == '-' else (a, other_cube.get_first(feature)))
    #     if isinstance(self.overlap(hypercubes), Container):
    #         raise Exception('Overlapping not handled')
    #
    # def expand_all(self, updates: Iterable[MinUpdate], surrounding: Container, ratio: float = 1.0) -> None:
    #     # todo: it won't work. Is it needed?
    #     for update in updates:
    #         self._expand_one(update, surrounding, ratio)
    #
    # def get_first(self, feature: str) -> float:
    #     # todo: it won't work. Is it needed?
    #     return self[feature][0]
    #
    # def get_second(self, feature: str) -> float:
    #     # todo: it won't work. Is it needed?
    #     return self[feature][1]
    #
    # def has_volume(self) -> bool:
    #     # todo: it won't work. Is it needed?
    #     return all([dimension[1] - dimension[0] > Container.EPSILON for dimension in self._constraints.values()])
    #
    # def volume(self) -> float:
    #     # todo: it won't work. Is it needed?
    #     return reduce(lambda a, b: a * b, [dimension[1] - dimension[0] for dimension in self._constraints.values()], 1)
    #
    # def diagonal(self) -> float:
    #     # todo: it won't work. Is it needed?
    #     return reduce(
    #         lambda a, b: a + b, [(dimension[1] - dimension[0]) ** 2 for dimension in self._constraints.values()], 0
    #     ) ** 0.5
    #
    # def is_adjacent(self, cont: Container) -> str | None:
    #     # todo: it won't work. Is it needed?
    #     return None
    #     # adjacent = None
    #     # for ((feature1, feature2), [a1, b1, c1]) in self._constraints.items():
    #     #     if self[feature] == cont[feature]:
    #     #         continue
    #     #     [a2, b2] = cont[feature]
    #     #     if (adjacent is not None) or ((b1 != a2) and (b2 != a1)):
    #     #         return None
    #     #     adjacent = feature
    #     # return adjacent
    #
    # def merge_along_dimension(self, cube: Container, feature: str) -> Container:
    #     # todo: it won't work. Is it needed?
    #     new_cube = self.copy()
    #     (a1, b1) = self[feature]
    #     (a2, b2) = cube[feature]
    #     new_cube.update_dimension(feature, (min(a1, a2), max(b1, b2)))
    #     return new_cube
    #
    # # TODO: maybe two different methods are more readable and easier to debug
    # def overlap(self, hypercubes: Iterable[Container] | Container) -> Container | bool | None:
    #     # todo: it won't work. Is it needed?
    #     if isinstance(hypercubes, Iterable):
    #         for hypercube in hypercubes:
    #             if (self != hypercube) & self.overlap(hypercube):
    #                 return hypercube
    #         return None
    #     elif self is hypercubes:
    #         return False
    #     else:
    #         return all([not ((dimension.other_dimension[0] >= dimension.this_dimension[1]) |
    #                          (dimension.this_dimension[0] >= dimension.other_dimension[1]))
    #                     for dimension in self._zip_dimensions(hypercubes)])
    #
    # # TODO: maybe two different methods are more readable and easier to debug
    # def update_dimension(self, feature: str, lower: float | tuple[float, float], upper: float | None = None) -> None:
    #     # todo: it won't work. Is it needed?
    #     if upper is None:
    #         self[feature] = lower
    #     else:
    #         self.update_dimension(feature, (lower, upper))
    #
    # def update(self, dataset: pd.DataFrame, predictor) -> None:
    #     # todo: it won't work. Is it needed?
    #     filtered = self._filter_dataframe(dataset.iloc[:, :-1])
    #     predictions = predictor.predict(filtered)
    #     self._output = np.mean(predictions)
    #     self._diversity = np.std(predictions)
    #
    # # TODO: why this is not a property?
    # def init_std(self, std: float) -> None:
    #     # todo: it won't work. Is it needed?
    #     self._diversity = std


GenericContainer = Union[Container]
