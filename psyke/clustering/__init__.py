from abc import ABC
from typing import Iterable

from psyke import Clustering, Target
from psyke.extraction.hypercubic import HyperCube, HyperCubePredictor


class HyperCubeClustering(HyperCubePredictor, Clustering, ABC):

    def __init__(self, output: Target = Target.CONSTANT, normalization=None):
        HyperCubePredictor.__init__(self, output=output, normalization=normalization)

    def get_hypercubes(self) -> Iterable[HyperCube]:
        raise NotImplementedError('predict')
