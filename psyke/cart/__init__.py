from psyke.cart.predictor import CartPredictor, LeafConstraints, LeafSequence
from psyke import Extractor, DiscreteFeature
from psyke.schema import GreaterThan
from psyke.utils.logic import create_variable_list, create_head, create_term
from tuprolog.core import clause, Var, Struct
from tuprolog.theory import Theory, mutable_theory
from typing import Iterable
import pandas as pd


class Cart(Extractor):

    def __init__(self, predictor: CartPredictor, discretization: Iterable[DiscreteFeature] = None,
                 simplify: bool = True):
        super().__init__(predictor, discretization)
        self.__simplify = simplify

    def __create_body(self, variables: dict[str, Var], constraints: LeafConstraints) -> Iterable[Struct]:
        results = []
        for name, value in constraints:
            features = [d for d in self.discretization if name in d.admissible_values]
            feature: DiscreteFeature = features[0] if len(features) > 0 else None
            results.append(create_term(variables[name], value) if feature is None else
                           create_term(variables[feature.name],
                                       feature.admissible_values[name],
                                       isinstance(value, GreaterThan)))
        return results

    def __create_theory(self, data: pd.DataFrame) -> Theory:
        new_theory = mutable_theory()
        for name, value in self.predictor:
            name = [(n[0], n[1]) for n in name if not self.__simplify or n[2]]
            variables = create_variable_list(self.discretization, data)
            new_theory.assertZ(
                clause(
                    create_head(data.columns[-1], list(variables.values()), value),
                    self.__create_body(variables, name)
                )
            )
        return new_theory

    def extract(self, data: pd.DataFrame) -> Theory:
        return self.__create_theory(data)

    def predict(self, data) -> Iterable:
        return self.predictor.predict(data)

    @property
    def n_rules(self) -> int:
        return self.predictor.n_leaves
