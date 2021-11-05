from __future__ import annotations
from tuprolog.theory import Theory
from psyke.schema.discrete_feature import DiscreteFeature


class Extractor:

    def __init__(self, predictor, discretization: set[DiscreteFeature] = None):
        self.predictor = predictor
        self.discretization = discretization

    def extract(self, dataset) -> Theory:
        pass

    def predict(self, dataset) -> list:
        pass

    @staticmethod
    def iter(predictor, min_update: float = 0.1, n_points: int = 1, max_iterations: int = 600, min_examples: int = 250,
             threshold: float = 0.1, fill_gaps: bool = False) -> Extractor:
        from psyke.regression.iter.iter import Iter
        return Iter(predictor, min_update, n_points, max_iterations, min_examples, threshold, fill_gaps)
