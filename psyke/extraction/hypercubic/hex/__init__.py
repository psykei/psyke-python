from itertools import product

import pandas as pd

from psyke import get_default_random_seed
from psyke.extraction.hypercubic import Grid, HyperCube
from psyke.extraction.hypercubic.gridex import GridEx


class HEx(GridEx):
    """
    Explanator implementing HEx algorithm.
    """

    def __init__(self, predictor, grid: Grid, min_examples: int, threshold: float, normalization,
                 seed=get_default_random_seed()):
        super().__init__(predictor, grid, min_examples, threshold, normalization, seed)
