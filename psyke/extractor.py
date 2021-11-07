from __future__ import annotations
from tuprolog.theory import Theory
from psyke.schema.discrete_feature import DiscreteFeature


class Extractor(object):
    """
    An explanator capable of extracting rules from trained black box.

    Parameters
    ----------
    predictor : the underling black box predictor.
    discretization : A collection of sets of discretised features.
        Each set corresponds to a set of features derived from a single non-discrete feature.
    """

    def __init__(self, predictor, discretization: set[DiscreteFeature] = None):
        self.predictor = predictor
        self.discretization = discretization

    def extract(self, dataset) -> Theory:
        """
        Extracts rules from the underlying predictor.

        :param dataset: is the set of instances to be used for the extraction.
        :return: the theory created from the extracted rules.
        """
        raise NotImplementedError('extract')

    def predict(self, dataset) -> list:
        """
        Predicts the output values of every sample in dataset.

        :param dataset: is the set of instances to predict.
        :return: a list of predictions.
        """
        raise NotImplementedError('predict')

    @staticmethod
    def iter(predictor, min_update: float = 0.1, n_points: int = 1, max_iterations: int = 600, min_examples: int = 250,
             threshold: float = 0.1, fill_gaps: bool = False) -> Extractor:
        """
        Creates a new Iter extractor.
        """
        from psyke.regression.iter.iter import ITER
        return ITER(predictor, min_update, n_points, max_iterations, min_examples, threshold, fill_gaps)
