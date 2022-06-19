from typing import Union

import pandas as pd
from tuprolog.theory import Theory

from psyke.regression import HyperCubeExtractor
from psyke.regression.hypercube import ClosedClassificationCube, ClosedCube, ClosedRegressionCube


class ClusterExtractor(HyperCubeExtractor):

    def __init__(self, predictor, depth: int, error_threshold: float,
                 output: HyperCubeExtractor.Target = HyperCubeExtractor.Target.CONSTANT, gauss_components: int = 2):
        super().__init__(predictor)
        self.depth = depth
        self.error_threshold = error_threshold
        self.gauss_components = gauss_components
        self._output = output

    def extract(self, dataframe: pd.DataFrame) -> Theory:
        raise NotImplementedError('extract')

    def _default_cube(self) -> Union[ClosedCube, ClosedRegressionCube, ClosedClassificationCube]:
        if self._output == ClusterExtractor.Target.CONSTANT:
            return ClosedCube()
        if self._output == ClusterExtractor.Target.REGRESSION:
            return ClosedRegressionCube()
        return ClosedClassificationCube()
