from abc import ABC

import pandas as pd
from tuprolog.theory import Theory

from psyke import Extractor


class PedagogicalExtractor(Extractor, ABC):

    def __init__(self, predictor, discretization=None, normalization=None):
        Extractor.__init__(self, predictor=predictor, discretization=discretization, normalization=normalization)

    def extract(self, dataframe: pd.DataFrame) -> Theory:
        new_y = pd.DataFrame(self.predictor.predict(dataframe.iloc[:, :-1])).set_index(dataframe.index)
        data = dataframe.iloc[:, :-1].copy().join(new_y)
        data.columns = dataframe.columns
        return self._extract(data)

    def _extract(self, dataframe: pd.DataFrame) -> Theory:
        raise NotImplementedError('extract')
