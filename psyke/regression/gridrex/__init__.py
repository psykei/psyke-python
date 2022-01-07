import pandas as pd
from sklearn.linear_model import LinearRegression
from tuprolog.core import Var

from psyke import get_default_random_seed
from psyke.regression import Grid
from psyke.regression.gridex import GridEx
from psyke.regression.hypercube import RegressionCube
from psyke.utils.logic import to_rounded_real, linear_function_creator


class GridREx(GridEx):
    """
    Explanator implementing GridREx algorithm.
    """

    def __init__(self, predictor, grid: Grid, min_examples: int, threshold: float, seed=get_default_random_seed()):
        super().__init__(predictor, grid, min_examples, threshold, seed)

    def _default_cube(self) -> RegressionCube:
        return RegressionCube()

    def _get_cube_output(self, cube: RegressionCube, data: dict[str, float]) -> float:
        return cube.output.predict(pd.DataFrame([data])).flatten()[0]

    def _create_output(self, variables: list[Var], output: LinearRegression):
        intercept = output.intercept_
        # intercept = self.__intercept(intercept, coefs) if self.scaler else intercept
        # coefs = [
        #     toRoundedReal(v / self.scaler.var_[f]**0.5) for (f, v) in enumerate(coefs)
        # ] if self.scaler else [toRoundedReal(v) for v in coefs]
        coefs = [to_rounded_real(v) for v in output.coef_]
        return linear_function_creator(variables, coefs, to_rounded_real(intercept))
