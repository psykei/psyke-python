from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from numpy import ndarray

from tuprolog.core import Var, Struct
from tuprolog.core import struct, real

from psyke.extraction.hypercubic import Node
from psyke.extraction.hypercubic.hypercube import ClosedClassificationCube
from psyke.extraction.hypercubic.hypercube import HyperCube
from psyke.schema import Between
from psyke.utils import get_default_precision, get_int_precision
from psyke.utils.logic import create_term, PRECISION


class OutputNotDefinedException(Exception):
    def __init__(self):
        super().__init__('The output of the container is not defined')


class ContainerNode(Node):
    def __init__(self, dataframe: pd.DataFrame, container: Container = None):
        super().__init__(dataframe, container)
        self.dataframe = dataframe
        self.container: Container = container
        self.right: ContainerNode | None = None
        self.left: ContainerNode | None = None

    def search(self, point: dict[str, float]) -> Container:
        if self.right is None:
            return self.container
        if self.right.container.__contains__(point):
            return self.right.search(point)
        return self.left.search(point)


class Container(ClosedClassificationCube):
    """
    A N-dimensional cube holding a numeric value.
    """

    EPSILON = get_default_precision()  # Precision used when comparing two hypercubes
    INT_PRECISION = get_int_precision()

    def __init__(self,
                 dimension: dict[str, tuple],
                 inequalities: dict[tuple[str, str], list[tuple[float, float, float]]] = {},
                 convex_hulls: Tuple = ([], 0)):
        """

        :param inequalities: is in the form (X,Y): a,b,c, which identifies the constraint aX + bY <= c,
            where X and Y are the names of the features that are being constrained
        """
        self._inequalities = inequalities
        self._output = None
        self.convex_hulls = convex_hulls
        super().__init__(dimension=dimension)

    def filter_indices(self, dataset: pd.DataFrame) -> ndarray:
        output = np.full(len(dataset.index), True, dtype=bool)
        for column in self.dimensions.keys():
            out = np.logical_and(dataset[column] >= self.dimensions[column][0], dataset[column] <= self.dimensions[column][1])
            if output is None:
                output = out
            else:
                output = np.logical_and(output, out)
        output = np.logical_and(output, Container.check_sat_constraints(self._inequalities, dataset))
        if isinstance(output, pd.Series):
            output = output.to_numpy()
        return output

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
        return Container(self.dimensions.copy(), self._inequalities.copy(), convex_hulls=self.convex_hulls)

    @property
    def inequalities(self) -> dict[tuple[str, str], list[tuple[float, float, float]]]:
        return self._inequalities

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
        constraints = self.inequalities.copy()
        for dimension in ignore:
            del dimensions[dimension]
        for (dim1, dim2) in self.inequalities:
            if dim1 in ignore or dim2 in ignore:
                del constraints[(dim1, dim2)]
        dimension_out = [create_term(variables[name], Between(unscale(values[0], name), unscale(values[1], name)))
                         for name, values in dimensions.items()]

        constr_out = []
        for dim1, dim2 in self.inequalities:
            for a, b, c in self.inequalities[(dim1, dim2)]:
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
        for X, Y in self.inequalities.keys():
            x = point[X]
            y = point[Y]
            for a, b, c in self._inequalities[(X, Y)]:
                if not(a * x + b * y <= c):
                    return False
        for dim in self.dimensions:
            start, end = self.dimensions[dim]
            if not(start <= point[dim] <= end):
                return False
        return True

    @property
    def output(self):
        if self._output is None:
            raise OutputNotDefinedException()
        return self._output

    def _fit_constraint(self, dimension: dict[tuple[str, str], tuple[float, float, float]]) -> dict[tuple[str, str], tuple[float, float, float]]:
        new_dimension: dict[tuple[str, str], tuple[float, float, float]] = {}
        for key, value in dimension.items():
            new_dimension[key] = (round(value[0], self.INT_PRECISION), round(value[1], self.INT_PRECISION), round(value[2], self.INT_PRECISION))
        return new_dimension
