import pandas as pd
from tuprolog.theory import Theory

from psyke.regression import HyperCubeExtractor


class GridEx(HyperCubeExtractor):
    """
    Explanator implementing GridEx algorithm, doi:10.1007/978-3-030-82017-6_2.
    """

    def extract(self, dataset: pd.DataFrame) -> Theory:
        pass
