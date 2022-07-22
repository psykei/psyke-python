from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from psyke.cart.predictor import CartPredictor, LeafConstraints, LeafSequence
from psyke import Extractor, DiscreteFeature, get_default_random_seed
from psyke.schema import GreaterThan
from psyke.utils.logic import create_variable_list, create_head, create_term
from tuprolog.core import clause, Var, Struct
from tuprolog.theory import Theory, mutable_theory
from typing import Iterable
import pandas as pd


CLASSIFICATION = 'classification'
REGRESSION = 'regression'
ADMISSIBLE_TASKS = (CLASSIFICATION, REGRESSION)
CART_PREDICTORS = {
    CLASSIFICATION: CartPredictor(DecisionTreeClassifier(random_state=get_default_random_seed())),
    REGRESSION: CartPredictor(DecisionTreeRegressor(max_depth=3, random_state=get_default_random_seed()))
}


class Cart(Extractor):

    def __init__(self, predictor, task: str = CLASSIFICATION, discretization: Iterable[DiscreteFeature] = None,
                 simplify: bool = True):
        super().__init__(predictor, discretization)
        if task in ADMISSIBLE_TASKS or task is None:
            self._cart_predictor = CART_PREDICTORS[task if task is not None else CLASSIFICATION]
        else:
            raise Exception("Wrong argument for task type. Accepted values are: " + ' '.join(ADMISSIBLE_TASKS))
        self._simplify = simplify

    def _create_body(self, variables: dict[str, Var], constraints: LeafConstraints) -> Iterable[Struct]:
        results = []
        for name, value in constraints:
            features = [d for d in self.discretization if name in d.admissible_values]
            feature: DiscreteFeature = features[0] if len(features) > 0 else None
            results.append(create_term(variables[name], value) if feature is None else
                           create_term(variables[feature.name],
                                       feature.admissible_values[name],
                                       isinstance(value, GreaterThan)))
        return results

    def _create_theory(self, data: pd.DataFrame) -> Theory:
        new_theory = mutable_theory()
        for name, value in self._cart_predictor:
            name = [(n[0], n[1]) for n in name if not self._simplify or n[2]]
            variables = create_variable_list(self.discretization, data)
            new_theory.assertZ(
                clause(
                    create_head(data.columns[-1], list(variables.values()), value),
                    self._create_body(variables, name)
                )
            )
        return new_theory

    def extract(self, data: pd.DataFrame) -> Theory:
        predicted_classes = self.predictor.predict(data.iloc[:, :-1])
        predicted_classes = pd.DataFrame(predicted_classes)
        predicted_classes.columns = [data.columns[-1]]
        new_data = data.iloc[:, :-1].join(predicted_classes)
        # If for any reason the predictor was not able to predict a class, ignore that data.
        new_data = new_data[new_data[new_data.columns[-1]].notnull()]
        self._cart_predictor.predictor.fit(new_data.iloc[:, :-1], new_data.iloc[:, -1])
        return self._create_theory(new_data)

    def predict(self, data) -> Iterable:
        return self._cart_predictor.predict(data)

    @property
    def n_rules(self) -> int:
        return self.predictor.n_leaves
