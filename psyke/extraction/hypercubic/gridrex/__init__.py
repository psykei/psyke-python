from psyke import get_default_random_seed, Target
from psyke.extraction.hypercubic import Grid, RegressionCube
from psyke.extraction.hypercubic.gridex import GridEx


class GridREx(GridEx):
    """
    Explanator implementing GridREx algorithm, doi:10.24963/kr.2022/57.
    """

    def __init__(self, predictor, grid: Grid, min_examples: int, threshold: float, normalization,
                 seed=get_default_random_seed()):
        super().__init__(predictor, grid, min_examples, threshold, Target.REGRESSION, None, normalization, seed)

    def _default_cube(self, dimensions=None) -> RegressionCube:
        return RegressionCube()
