from __future__ import annotations

from collections import Iterable
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from tuprolog.theory import Theory
from psyke import Clustering
from psyke.clustering import HyperCubeClustering
from psyke.extraction.hypercubic import HyperCubeExtractor
from psyke.utils import Target, get_default_random_seed


class CReEPy(HyperCubeExtractor):
    """
    Explanator implementing CReEPy algorithm.
    """

    def __init__(self, predictor, clustering=Clustering.exact, depth: int = 3, error_threshold: float = 0.1,
                 output: Target = Target.CONSTANT, gauss_components: int = 5, ranks: list[(str, float)] = [],
                 ignore_threshold: float = 0.0, discretization=None, normalization=None,
                 seed: int = get_default_random_seed()):
        super().__init__(predictor, Target.CLASSIFICATION if isinstance(predictor, ClassifierMixin) else output,
                         discretization, normalization)
        self.clustering = clustering(depth, error_threshold, self._output, gauss_components, discretization,
                                     normalization, seed)
        self._default_surrounding_cube = True
        self._dimensions_to_ignore = set([dimension for dimension, relevance in ranks if relevance < ignore_threshold])

    def _extract(self, dataframe: pd.DataFrame) -> Theory:
        if not isinstance(self.clustering, HyperCubeClustering):
            raise TypeError("clustering must be a HyperCubeClustering")

        self.clustering.fit(dataframe)
        self._hypercubes = self.clustering.get_hypercubes()
        self._surrounding = self._hypercubes[-1]
        return self._create_theory(dataframe)
