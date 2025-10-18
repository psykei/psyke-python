from typing import Iterable

import numpy as np
import pandas as pd
from tuprolog.theory import Theory

from psyke import get_default_random_seed
from psyke.extraction.hypercubic import HyperCubeExtractor


class GInGER(HyperCubeExtractor):
    """
    Explanator implementing GInGER algorithm.
    """

    def __init__(self, predictor, normalization=None, seed: int = get_default_random_seed()):
        super().__init__(predictor, normalization)
        np.random.seed(seed)

    def _extract(self, dataframe: pd.DataFrame) -> Theory:
        #self._hypercubes = []
        #self._surrounding = HyperCube.create_surrounding_cube(dataframe, output=self._output)
        #self._surrounding.init_diversity(2 * self.threshold)
        #self._iterate(dataframe)
        return self._create_theory(dataframe)

    def make_fair(self, features: Iterable[str]):
        self._dimensions_to_ignore.update(features)
