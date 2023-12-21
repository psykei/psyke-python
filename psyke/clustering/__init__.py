from abc import ABC
from typing import Iterable

from psyke import Clustering, Target
from psyke.extraction.hypercubic import HyperCube
from psyke.hypercubepredictor import HyperCubePredictor


class HyperCubeClustering(HyperCubePredictor, Clustering, ABC):

    def __init__(self, output: Target = Target.CONSTANT, discretization=None, normalization=None):
        HyperCubePredictor.__init__(self, output=output, discretization=discretization, normalization=normalization)

    def get_hypercubes(self) -> Iterable[HyperCube]:
        raise NotImplementedError('get_hypercubes')
