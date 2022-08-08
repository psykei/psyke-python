from typing import Union

import pandas as pd
from tuprolog.theory import Theory

from psyke.extraction.hypercubic.hypercube import ClosedRegressionCube, ClosedClassificationCube, ClosedCube
from psyke.utils import Target


class InterpretableClustering:

    def __init__(self, depth: int, error_threshold: float, output: Target = Target.CONSTANT, gauss_components: int = 2):
        self.depth = depth
        self.error_threshold = error_threshold
        self.gauss_components = gauss_components
        self._output = output
        self._hypercubes = []

    def extract(self, dataframe: pd.DataFrame) -> Theory:
        raise NotImplementedError('extract')

    def _default_cube(self) -> Union[ClosedCube, ClosedRegressionCube, ClosedClassificationCube]:
        if self._output == Target.CONSTANT:
            return ClosedCube()
        if self._output == Target.REGRESSION:
            return ClosedRegressionCube()
        return ClosedClassificationCube()
