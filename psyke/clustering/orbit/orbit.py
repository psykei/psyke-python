from __future__ import annotations
from sklearn.base import ClassifierMixin

from psyke import Extractor
from psyke.extraction.hypercubic.creepy import CReEPy
from psyke.utils import Target
from collections import Iterable
import numpy as np
import pandas as pd
from tuprolog.core import clause
from tuprolog.theory import Theory

from psyke import Extractor
from psyke.utils import Target
from psyke.clustering.orbit.constraint_generator import ConstraintGenerator
from tuprolog.theory import Theory, mutable_theory
from psyke import Extractor, logger
from psyke.utils.logic import create_variable_list, create_head, to_var
from tuprolog.core import Var, Struct, clause
from sklearn.linear_model import LinearRegression
from psyke.clustering.orbit.container import Container
from typing import List, Tuple
from psyke.utils import Target, get_int_precision

class ORBIt(Extractor):
    """
    Oblique Rule-Based ITerative clustering
    """

    def __init__(self, predictor, depth: int, error_threshold: float,
                 gauss_components: int = 5, ranks: list[(str, float)] = [], ignore_threshold: float = 0.0,
                 normalization=None, steps=1000, min_accuracy_increase=0.01, max_disequation_num=4):
        """

        :param predictor: object that must contain a function predict in order to predict the cluster label, given a dataframe
        :param depth: depth of the tree of rules (contraints) that will be generated
        :param error_threshold:
        :param gauss_components: number of gaussian clusters used to split data into different hyper-cubes/oblique rules
        :param ranks: kept for coherency with orchid
        :param ignore_threshold: kept for coherency with orchid
        :param normalization: kept for coherency with orchid
        :param steps: every time disequations are created,
            only steps couples of dimensions are checked to generate disequations
        :param min_accuracy_increase: oblique rules (diequtions) are preferred to hypercubes only
            if it causes an increse in total accuracy of "min_accuracy_increase".
            if min_accuracy_increase is 0, it prefers oblique rules only if there is an increase (even minimal) of
                accuracy.
            if min_accuracy_increase is less than 0, it always prefers oblique rules.
        :param max_disequation_num: maximum number of disequations for each couple of dimension
        """
        super().__init__(predictor, normalization=normalization)
        self.gauss_components = gauss_components
        self.depth = depth
        self.error_threshold = error_threshold
        self.ranks = ranks
        self.ignore_threshold = ignore_threshold
        self.clustering = ConstraintGenerator(depth, error_threshold, gauss_components, steps,
                                              min_accuracy_increase, max_disequation_num=max_disequation_num)
        self.containers: List[Container] = []

    def extract(self, dataframe: pd.DataFrame) -> Theory:
        """
        extract theory out of a dataframe
        :param dataframe: dataframe containing as the last column the label representing the clusters
        :return:
        """
        self.containers = self.clustering.extract(dataframe=dataframe.iloc[:, :-1].join(
            pd.DataFrame(self.predictor.predict(dataframe.iloc[:, :-1]), index=dataframe.index)
        ))
        # # is this part needed?
        # for container in self.containers:
        #     for dimension in self._ignore_dimensions():
        #         container[dimension] = [-np.inf, np.inf]
        theory, disequations = self._create_theory(dataframe)
        last_clause = list(theory.clauses)[-1]
        theory.retract(last_clause)
        theory.assertZ(clause(last_clause.head, []))
        last_cube = self.containers[-1]
        for dimension in last_cube.dimensions.keys():
            last_cube[dimension] = [-np.inf, np.inf]
        return theory   #, [c.convex_hulls for c in self.containers]

    def _ignore_dimensions(self) -> List[str]:
        return [dimension for dimension, relevance in self.ranks if relevance < self.ignore_threshold]

    @property
    def n_rules(self):
        return len(list(self.containers))

    @property
    def n_oblique_rules(self):
        return sum([len(c.diequations) for c in self.containers])

    def _create_theory(self, dataframe: pd.DataFrame) -> Theory:
        new_theory = mutable_theory()
        disequations = []
        for cube in self.containers:
            logger.info(cube.output)
            logger.info(cube.dimensions)
            variables = create_variable_list([], dataframe)
            variables[dataframe.columns[-1]] = to_var(dataframe.columns[-1])
            head = ORBIt._create_head(dataframe, list(variables.values()),
                                                   self.unscale(cube.output, dataframe.columns[-1]))
            body = cube.body(variables, self._ignore_dimensions(), self.unscale, self.normalization)
            new_theory.assertZ(clause(head, body))
            disequations.append((cube.output, cube.diequations))
        return new_theory, disequations

    @staticmethod
    def _create_head(dataframe: pd.DataFrame, variables: list[Var], output: float | LinearRegression) -> Struct:
        return create_head(dataframe.columns[-1], variables[:-1], output)

    def unscale(self, values, name):
        if self.normalization is None or isinstance(values, LinearRegression):
            return values
        if isinstance(values, Iterable):
            idx = [value is not None for value in values]
            values[idx] = values[idx] * self.normalization[name][1] + self.normalization[name][0]
        else:
            values = values * self.normalization[name][1] + self.normalization[name][0]
        return values

    def predict(self, dataframe: pd.DataFrame) -> Iterable:
        return np.array([self._predict(dict(row.to_dict())) for _, row in dataframe.iterrows()])

    def _predict(self, data: dict[str, float]) -> float | None:
        data = {k: v for k, v in data.items()}
        for container in self.containers:
            if container.__contains__(data):
                return round(container.output, get_int_precision())
        return None

    # @staticmethod
    # def _get_cube_output(cube: HyperCube | RegressionCube, data: dict[str, float]) -> float:
    #     return cube.output.predict(pd.DataFrame([data])).flatten()[0] if \
    #         isinstance(cube, RegressionCube) else cube.output