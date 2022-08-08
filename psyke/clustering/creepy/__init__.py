from __future__ import annotations

from collections import Iterable

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from tuprolog.theory import Theory

from psyke import Extractor
from psyke.regression import HyperCubeExtractor
from psyke.utils import Target


class CReEPy(HyperCubeExtractor):
    """
    Explanator implementing CReEPy algorithm.
    """

    def __init__(self, predictor, depth: int, error_threshold: float, output: Target = Target.CONSTANT,
                 gauss_components: int = 5, ranks: list[(str, float)] = [], ignore_threshold: float = 0.0):
        super().__init__(predictor)
        self.clustering = Extractor.exact(depth, error_threshold, Target.CLASSIFICATION if
                                          isinstance(predictor, ClassifierMixin) else output, gauss_components)
        self._output = output
        self.ranks = ranks
        self.ignore_threshold = ignore_threshold

    def extract(self, dataframe: pd.DataFrame) -> Theory:
        self._hypercubes = self.clustering.extract(dataframe.iloc[:, :-1].join(
            pd.DataFrame(self.predictor.predict(dataframe.iloc[:, :-1]), index=dataframe.index)
        ))
        for cube in self._hypercubes:
            for dimension in self._ignore_dimensions():
                cube[dimension] = [-np.inf, np.inf]
        return self._create_theory(dataframe)

    def _ignore_dimensions(self) -> Iterable[str]:
        return [dimension for dimension, relevance in self.ranks if relevance < self.ignore_threshold]

    @property
    def n_rules(self):
        return len(list(self._hypercubes))
