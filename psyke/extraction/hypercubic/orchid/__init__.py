from __future__ import annotations
from psyke import Extractor
from psyke.extraction.hypercubic.creepy import CReEPy
from psyke.utils import Target


class ORCHiD(CReEPy):
    """
    Explanator implementing ORCHiD algorithm.
    """

    def __init__(self, predictor, depth: int, error_threshold: float, output: Target = Target.CONSTANT,
                 gauss_components: int = 5, ranks: list[(str, float)] = [], ignore_threshold: float = 0.0,
                 normalization=None):
        super().__init__(predictor, depth, error_threshold, output, gauss_components, ranks, ignore_threshold,
                         normalization, clustering=Extractor.cream)
