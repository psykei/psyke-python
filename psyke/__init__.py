from __future__ import annotations
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

    def __init__(self, predictor, discretization: Iterable[DiscreteFeature] = None):
        self.predictor = predictor
        self.discretization = [] if discretization is None else list(discretization)

    def extract(self, dataset) -> Theory:
        """
        Extracts rules from the underlying predictor.

        :param dataset: is the set of instances to be used for the extraction.
        :return: the theory created from the extracted rules.
        """
        raise NotImplementedError('extract')

    def predict(self, dataset) -> Iterable:
        """
        Predicts the output values of every sample in dataset.

        :param dataset: is the set of instances to predict.
        :return: a list of predictions.
        """
        raise NotImplementedError('predict')

    @staticmethod
    def cart(predictor: cart.CartPredictor, discretization=None) -> Extractor:
        """
        Creates a new Cart extractor.
        """
        from psyke.cart import Cart
        return Cart(predictor, discretization)

    @staticmethod
    def iter(predictor, min_update: float = 0.1, n_points: int = 1, max_iterations: int = 600, min_examples: int = 250,
             threshold: float = 0.1, fill_gaps: bool = False, seed: int = get_default_random_seed()) -> Extractor:
        """
        Creates a new ITER extractor.
        """
        from psyke.regression.iter import ITER
        return ITER(predictor, min_update, n_points, max_iterations, min_examples, threshold, fill_gaps, seed)

    @staticmethod
    def real(predictor, discretization=None) -> Extractor:
        """
        Creates a new REAL extractor.
        """
        from psyke.classification.real import REAL
        return REAL(predictor, [] if discretization is None else discretization)

    @staticmethod
    def trepan(predictor, discretization=None, min_examples: int = 0, max_depth: int = 3,
               split_logic=None) -> Extractor:
        """
        Creates a new Trepan extractor.
        """
        from psyke.classification.trepan import Trepan, SplitLogic
        if split_logic is None:
            split_logic = SplitLogic.DEFAULT
        return Trepan(predictor, [] if discretization is None else discretization, min_examples, max_depth, split_logic)