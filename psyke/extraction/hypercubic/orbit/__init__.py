from __future__ import annotations

from collections import Iterable
import numpy as np
import pandas as pd
from psyke.extraction.hypercubic.orbit.mixed_rules_extractor import MixedRulesExtractor
from tuprolog.theory import Theory, mutable_theory
from psyke import logger, Target
from psyke.extraction.hypercubic import HyperCubeExtractor
from psyke.utils.logic import create_variable_list, to_var
from tuprolog.core import clause
from psyke.extraction.hypercubic.orbit.container import Container
from typing import List
from psyke.utils import get_int_precision


class ORBIt(HyperCubeExtractor):
    """
    Oblique Rule-Based ITerative clustering
    """

    def __init__(self, predictor, depth: int, error_threshold: float,
                 gauss_components: int = 5, output: Target = Target.CLASSIFICATION,
                 ranks: list[(str, float)] = [], ignore_threshold: float = 0.0,
                 normalization=None, steps=1000, min_accuracy_increase=0.01, max_disequation_num=4):
        """

        :param predictor: object that must contain a function predict in order to predict the cluster label,
            given a dataframe
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
        super().__init__(predictor, output=output, normalization=normalization)
        self.gauss_components = gauss_components
        self.depth = depth
        self.error_threshold = error_threshold
        self.ranks = ranks
        self.ignore_threshold = ignore_threshold
        self.clustering = MixedRulesExtractor(depth, error_threshold, gauss_components, steps,
                                              min_accuracy_increase, max_disequation_num, output)
        self._hypercubes: List[Container] = []

    def _extract(self, dataframe: pd.DataFrame, mapping: dict[str: int] = None, sort: bool = True) -> Theory:
        """
        extract theory out of a dataframe
        :param dataframe: dataframe containing as the last column the label representing the clusters
        :return:
        """
        self._hypercubes = self.clustering.extract(dataframe=dataframe.iloc[:, :-1].join(
            pd.DataFrame(self.predictor.predict(dataframe.iloc[:, :-1]), index=dataframe.index)
        ))
        theory, disequations = self._create_theory(dataframe)
        last_clause = list(theory.clauses)[-1]
        theory.retract(last_clause)
        theory.assertZ(clause(last_clause.head, []))
        last_cube = self._hypercubes[-1]
        for dimension in last_cube.dimensions.keys():
            last_cube[dimension] = [-np.inf, np.inf]
        return theory

    def _create_theory(self, dataframe: pd.DataFrame, sort: bool = True) -> Theory:
        new_theory = mutable_theory()
        disequations = []
        for cube in self._hypercubes:
            logger.info(cube.output)
            logger.info(cube.dimensions)
            variables = create_variable_list([], dataframe)
            variables[dataframe.columns[-1]] = to_var(dataframe.columns[-1])
            head = HyperCubeExtractor._create_head(dataframe, list(variables.values()),
                                                   self.unscale(cube.output, dataframe.columns[-1]))
            body = cube.body(variables, self._ignore_dimensions(), self.unscale, self.normalization)
            new_theory.assertZ(clause(head, body))
            disequations.append((cube.output, cube.inequalities))
        return HyperCubeExtractor._prettify_theory(new_theory), disequations

    def _ignore_dimensions(self) -> List[str]:
        return [dimension for dimension, relevance in self.ranks if relevance < self.ignore_threshold]

    @property
    def n_oblique_rules(self):
        return sum([len(c.inequalities) for c in self._hypercubes])

    def _predict(self, dataframe: pd.DataFrame) -> Iterable:
        rows = [dict(row.to_dict()) for _, row in dataframe.iterrows()]
        predictions = []
        for data in rows:
            prediction = None
            data = {k: v for k, v in data.items()}
            for container in self._hypercubes:
                if data in container:
                    prediction = round(container.output, get_int_precision())
                    break
            predictions.append(prediction)
        return np.array(predictions)
