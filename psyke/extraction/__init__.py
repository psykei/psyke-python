from abc import ABC
from collections import Iterable

import pandas as pd
from tuprolog.theory import Theory

from psyke import Extractor


class PedagogicalExtractor(Extractor, ABC):

    def __init__(self, predictor, discretization=None, normalization=None):
        Extractor.__init__(self, predictor=predictor, discretization=discretization, normalization=normalization)

    def _substitute_output(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        new_y = pd.DataFrame(self.predictor.predict(dataframe.iloc[:, :-1])).set_index(dataframe.index)
        data = dataframe.iloc[:, :-1].copy().join(new_y)
        data.columns = dataframe.columns
        return data

    def extract(self, dataframe: pd.DataFrame) -> Theory:
        self.theory = self._extract(self._substitute_output(dataframe))
        return self.theory

    def _extract(self, dataframe: pd.DataFrame) -> Theory:
        raise NotImplementedError('extract')


class FairExtractor(PedagogicalExtractor, ABC):

    def __init__(self, extractor: Extractor, features: Iterable):
        super().__init__(extractor.predictor, extractor.discretization, extractor.normalization)
        self.features = features
        self.extractor = extractor
        # self.make_fair()

#    def extract(self, dataframe: pd.DataFrame) -> Theory:
#        self.theory = self.extractor.extract(dataframe)
#        return self.theory

#    def predict_why(self, data: dict[str, float], verbose: bool = True):
#        self.extractor.predict_why(data, verbose)

#    def predict_counter(self, data: dict[str, float], verbose: bool = True, only_first: bool = True):
#        self.extractor.predict_counter(data, verbose, only_first)

#    def _predict(self, dataframe: pd.DataFrame) -> Iterable:
#        return self.extractor.predict(dataframe)

#    def _brute_predict(self, dataframe: pd.DataFrame, criterion: str = 'corner', n: int = 2) -> Iterable:
#        return self.extractor.brute_predict(dataframe, criterion, n)
