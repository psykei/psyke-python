from __future__ import annotations
from abc import ABC
import numpy as np
import pandas as pd
from numpy import argmax
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, accuracy_score
from psyke.schema import DiscreteFeature
from psyke.utils import get_default_random_seed
from tuprolog.theory import Theory
from typing import Iterable
import logging


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('psyke')


class Extractor(object):
    """
    An explanator capable of extracting rules from trained black box.

    Parameters
    ----------
    predictor : the underling black box predictor.
    discretization : A collection of sets of discretised features.
        Each set corresponds to a set of features derived from a single non-discrete feature.
    """

    def __init__(self, predictor, discretization: Iterable[DiscreteFeature] = None, normalization=None):
        self.predictor = predictor
        self.discretization = [] if discretization is None else list(discretization)
        self.normalization = normalization

    def extract(self, dataframe: pd.DataFrame, mapping: dict[str: int] = None, sort: bool = True) -> Theory:
        """
        Extracts rules from the underlying predictor.

        :param dataframe: is the set of instances to be used for the extraction.
        :param mapping: for one-hot encoding.
        :param sort: alphabetically sort the variables of the head of the rules.
        :return: the theory created from the extracted rules.
        """
        raise NotImplementedError('extract')

    def predict(self, dataframe: pd.DataFrame, mapping: dict[str: int] = None) -> Iterable:
        """
        Predicts the output values of every sample in dataset.

        :param dataframe: is the set of instances to predict.
        :param mapping: for one-hot encoding.
        :return: a list of predictions.
        """
        ys = self._predict(dataframe)
        if mapping is not None:
            inverse_mapping = {v: k for k, v in mapping.items()}
            ys = [inverse_mapping[y] for y in ys]
        return ys

    def _predict(self, dataframe: pd.DataFrame) -> Iterable:
        raise NotImplementedError('predict')

    def unscale(self, values, name):
        if self.normalization is None or isinstance(values, LinearRegression):
            return values
        if isinstance(values, Iterable):
            idx = [value is not None for value in values]
            values[idx] = values[idx] * self.normalization[name][1] + self.normalization[name][0]
        else:
            values = values * self.normalization[name][1] + self.normalization[name][0]
        return values

    def regression_score(self, dataframe: pd.DataFrame, predictor=None, scoring_function=mean_absolute_error):
        predictions = np.array(self.predict(dataframe.iloc[:, :-1]))
        idx = [prediction is not None for prediction in predictions]
        true = self.unscale(dataframe.iloc[idx, -1] if predictor is None else
                            predictor.predict(dataframe.iloc[idx, :-1]).flatten(), dataframe.columns[-1])
        return scoring_function(true, self.unscale(predictions[idx], dataframe.columns[-1]))

    def mae(self, dataframe: pd.DataFrame, predictor=None) -> float:
        """
        Calculates the predictions' MAE w.r.t. the instances given as input.

        :param dataframe: is the set of instances to be used to calculate the mean absolute error.
        :param predictor: if provided, its predictions on the dataframe are taken instead of the dataframe instances.
        :return: the mean absolute error (MAE) of the predictions.
        """
        return self.regression_score(dataframe, predictor, mean_absolute_error)

    def mse(self, dataframe: pd.DataFrame, predictor=None) -> float:
        """
        Calculates the predictions' MSE w.r.t. the instances given as input.

        :param dataframe: is the set of instances to be used to calculate the mean squared error.
        :param predictor: if provided, its predictions on the dataframe are taken instead of the dataframe instances.
        :return: the mean squared error (MSE) of the predictions.
        """
        return self.regression_score(dataframe, predictor, mean_squared_error)

    def r2(self, dataframe: pd.DataFrame, predictor=None) -> float:
        """
        Calculates the predictions' R2 score w.r.t. the instances given as input.

        :param dataframe: is the set of instances to be used to calculate the R2 score.
        :param predictor: if provided, its predictions on the dataframe are taken instead of the dataframe instances.
        :return: the R2 score of the predictions.
        """
        return self.regression_score(dataframe, predictor, r2_score)

    def accuracy(self, dataframe: pd.DataFrame, predictor=None) -> float:
        """
        Calculates the predictions' accuracy classification score w.r.t. the instances given as input.

        :param dataframe: is the set of instances to be used to calculate the accuracy classification score.
        :param predictor: if provided, its predictions on the dataframe are taken instead of the dataframe instances.
        :return: the accuracy classification score of the predictions.
        """
        predictions = np.array(self.predict(dataframe.iloc[:, :-1]))
        idx = [prediction is not None for prediction in predictions]
        return accuracy_score(dataframe.iloc[idx, -1] if predictor is None else
                              predictor.predict(dataframe.iloc[idx, :-1]).flatten(),
                              predictions[idx])

    def f1(self, dataframe: pd.DataFrame, predictor=None) -> float:
        """
        Calculates the predictions' F1 score w.r.t. the instances given as input.

        :param dataframe: is the set of instances to be used to calculate the F1 score.
        :param predictor: if provided, its predictions on the dataframe are taken instead of the dataframe instances.
        :return: the F1 score of the predictions.
        """
        predictions = np.array(self.predict(dataframe.iloc[:, :-1])[:])
        idx = [prediction is not None for prediction in predictions]
        return f1_score(dataframe.iloc[idx, -1] if predictor is None else
                        predictor.predict(dataframe.iloc[idx, :-1]).flatten(),
                        predictions[idx], average='weighted')

    @staticmethod
    def exact(depth: int, error_threshold: float, output, gauss_components: int = 2):
        """
        Creates a new ExACT instance.
        """
        from psyke.clustering.exact import ExACT
        return ExACT(depth, error_threshold, output, gauss_components)

    @staticmethod
    def cream(depth: int, error_threshold: float, output, gauss_components: int = 2):
        """
        Creates a new CREAM instance.
        """
        from psyke.clustering.cream import CREAM
        return CREAM(depth, error_threshold, output, gauss_components)

    @staticmethod
    def cart(predictor, max_depth: int = 3, max_leaves: int = 3,
             discretization: Iterable[DiscreteFeature] = None, normalization=None, simplify: bool = True) -> Extractor:
        """
        Creates a new Cart extractor.
        """
        from psyke.extraction.cart import Cart
        return Cart(predictor, max_depth, max_leaves, discretization=discretization, normalization=normalization,
                    simplify=simplify)

    @staticmethod
    def iter(predictor, min_update: float = 0.1, n_points: int = 1, max_iterations: int = 600, min_examples: int = 250,
             threshold: float = 0.1, fill_gaps: bool = True, normalization: dict[str, tuple[float, float]] = None,
             output=None, seed: int = get_default_random_seed()) -> Extractor:
        """
        Creates a new ITER extractor.
        """
        from psyke.extraction.hypercubic.iter import ITER
        return ITER(predictor, min_update, n_points, max_iterations, min_examples, threshold, fill_gaps,
                    normalization, output, seed)

    @staticmethod
    def gridex(predictor, grid, min_examples: int = 250, threshold: float = 0.1,
               normalization: dict[str, tuple[float, float]] = None,
               seed: int = get_default_random_seed()) -> Extractor:
        """
        Creates a new GridEx extractor.
        """
        from psyke.extraction.hypercubic.gridex import GridEx
        return GridEx(predictor, grid, min_examples, threshold, normalization, seed)

    @staticmethod
    def gridrex(predictor, grid, min_examples: int = 250, threshold: float = 0.1,
                normalization: dict[str, tuple[float, float]] = None,
                seed: int = get_default_random_seed()) -> Extractor:
        """
        Creates a new GridREx extractor.
        """
        from psyke.extraction.hypercubic.gridrex import GridREx
        return GridREx(predictor, grid, min_examples, threshold, normalization, seed)

    @staticmethod
    def creepy(predictor, depth: int, error_threshold: float, output, gauss_components: int = 2,
               ranks: [(str, float)] = [], ignore_threshold: float = 0.0,
               normalization: dict[str, tuple[float, float]] = None, clustering=exact) -> Extractor:
        """
        Creates a new CReEPy extractor.
        """
        from psyke.extraction.hypercubic.creepy import CReEPy
        return CReEPy(predictor, depth, error_threshold, output, gauss_components, ranks, ignore_threshold,
                      normalization, clustering)

    @staticmethod
    def real(predictor, discretization=None) -> Extractor:
        """
        Creates a new REAL extractor.
        """
        from psyke.extraction.real import REAL
        return REAL(predictor, [] if discretization is None else discretization)

    @staticmethod
    def trepan(predictor, discretization=None, min_examples: int = 0, max_depth: int = 3,
               split_logic=None) -> Extractor:
        """
        Creates a new Trepan extractor.
        """
        from psyke.extraction.trepan import Trepan, SplitLogic
        if split_logic is None:
            split_logic = SplitLogic.DEFAULT
        return Trepan(predictor, [] if discretization is None else discretization, min_examples, max_depth, split_logic)


class PedagogicalExtractor(Extractor, ABC):

    def extract(self, dataframe: pd.DataFrame, mapping: dict[str: int] = None, sort: bool = True) -> Theory:
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
        raise NotImplementedError('predict')
