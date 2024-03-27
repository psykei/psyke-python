import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from tuprolog.theory import Theory

from psyke import Target, Extractor, get_default_random_seed
from psyke.clustering.utils import select_gaussian_mixture
from psyke.extraction.hypercubic import HyperCube, HyperCubeExtractor, RegressionCube


class COSMiK(HyperCubeExtractor):
    """
    Explanator implementing COSMiK algorithm.
    """

    def __init__(self, predictor, max_components: int = 4, k: int = 5, patience: int = 15, close_to_center: bool = True,
                 output: Target = Target.CONSTANT, discretization=None, normalization=None,
                 seed: int = get_default_random_seed()):
        super().__init__(predictor, Target.REGRESSION, discretization, normalization)
        self.max = max_components
        self.k = k
        self.patience = patience
        self.output = output
        self.close_to_center = close_to_center
        self.seed = seed

    def _extract(self, dataframe: pd.DataFrame) -> Theory:
        np.random.seed(self.seed)
        X, y = dataframe.iloc[:, :-1], dataframe.iloc[:, -1]

        _, n, _ = select_gaussian_mixture(dataframe, self.max)
        gmm = GaussianMixture(n)
        gmm.fit(X, y)

        divine = Extractor.divine(gmm, self.k, self.patience, self.close_to_center,
                                  self.discretization, self.normalization)
        df = X.join(pd.DataFrame(gmm.predict(X)))
        df.columns = dataframe.columns
        divine.extract(df)

        self._hypercubes = [HyperCube(cube.dimensions.copy()) if self.output == Target.CONSTANT else
                            RegressionCube(cube.dimensions.copy()) for cube in divine._hypercubes]
        for cube in self._hypercubes:
            cube.update(dataframe, self.predictor)

        self._sort_cubes()
        return self._create_theory(dataframe)