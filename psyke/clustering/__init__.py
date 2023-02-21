from abc import ABC
from typing import Iterable

from psyke import Clustering, HyperCubePredictor, Target
from psyke.extraction.hypercubic import HyperCube


class HyperCubeClustering(HyperCubePredictor, Clustering, ABC):

    def __init__(self, output: Target = Target.CONSTANT):
        HyperCubePredictor.__init__(self, output=output)

    def get_hypercubes(self) -> Iterable[HyperCube]:
        raise NotImplementedError('predict')
