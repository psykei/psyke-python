from abc import ABC

import pandas as pd
from numpy import argmax
from tuprolog.theory import Theory

from psyke import Extractor


class PedagogicalExtractor(Extractor, ABC):

    def __init__(self, predictor, discretization=None, normalization=None):
        Extractor.__init__(self, predictor=predictor, discretization=discretization, normalization=normalization)

    def extract(self, dataframe: pd.DataFrame, mapping: dict[str: int] = None, sort: bool = True) -> Theory:
        from psyke.extraction.hypercubic import HyperCubeExtractor, HyperCube
        new_y = self.predictor.predict(dataframe.iloc[:, :-1])
        if mapping is not None:
            if hasattr(new_y[0], 'shape'):
                # One-hot encoding for multi-class tasks
                if len(new_y[0].shape) > 0 and new_y[0].shape[0] > 1:
                    new_y = [argmax(y, axis=0) for y in new_y]
                # One-hot encoding for binary class tasks
                else:
                    new_y = [round(y[0]) for y in new_y]
        new_y = pd.DataFrame(new_y).set_index(dataframe.index)
        data = dataframe.iloc[:, :-1].copy().join(new_y)
        data.columns = dataframe.columns
        return self._extract(data, mapping, sort)

    def _extract(self, dataframe: pd.DataFrame, mapping: dict[str: int] = None, sort: bool = True) -> Theory:
        raise NotImplementedError('extract')
