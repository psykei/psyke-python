import pandas as pd
from tuprolog.theory import Theory

from psyke.regression import HyperCubeExtractor, ClosedCube, ClosedRegressionCube


class ClusterExtractor(HyperCubeExtractor):

    def __init__(self, predictor, depth: int, error_threshold: float,
                 gauss_components: int = 2, constant: bool = False):
        super().__init__(predictor)
        self.depth = depth
        self.error_threshold = error_threshold
        self.gauss_components = gauss_components
        self._constant = constant

    def extract(self, dataframe: pd.DataFrame) -> Theory:
        raise NotImplementedError('extract')

    def _default_cube(self) -> ClosedCube:
        return ClosedCube() if self._constant else ClosedRegressionCube()