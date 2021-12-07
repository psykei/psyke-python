from functools import lru_cache
from psyke.classification.real.utils import Rule, IndexedRuleSet
from psyke import Extractor
from psyke.schema import DiscreteFeature
from psyke.utils.dataframe import get_discrete_dataset, HashableDataFrame
from psyke.utils.logic import create_term, create_head, create_variable_list
from tuprolog.core import Var, Struct, Clause, clause
from tuprolog.theory import MutableTheory, mutable_theory, Theory
from typing import Iterable
import pandas as pd


class REAL(Extractor):
    """
    Explanator implementing Rule Extraction As Learning (REAL) algorithm, doi:10.1016/B978-1-55860-335-6.50013-1.
    The algorithm is sensible to features' order in the provided dataset during extraction.
    To make it reproducible the features are internally sorted (alphabetically).
    """

    def __init__(self, predictor, discretization: Iterable[DiscreteFeature]):
        super().__init__(predictor, discretization)
        self.__ruleset: IndexedRuleSet = IndexedRuleSet()
        self.__output_mapping = {}

    def __covers(self, sample: pd.Series, rules: list[Rule]) -> bool:
        new_rule = self.__rule_from_example(sample)
        return any([new_rule.is_sub_rule_of(rule) for rule in rules])

    def __create_body(self, variables: dict[str, Var], rule: Rule) -> list[Struct]:
        result = []
        for predicates, truth_value in zip(rule.to_lists(), [True, False]):
            for predicate in predicates:
                feature = [feature for feature in self.discretization if predicate in feature.admissible_values][0]
                result.append(create_term(variables[feature.name], feature.admissible_values[predicate], truth_value))
        return result

    def __create_clause(self, dataset: pd.DataFrame, variables: dict[str, Var], key: int, rule: Rule) -> Clause:
        head = create_head(dataset.columns[-1],
                           sorted(list(variables.values())),
                           str(sorted(list(set(dataset.iloc[:, -1])))[key]))
        return clause(head, self.__create_body(variables, rule))

    def __create_new_rule(self, sample: pd.Series) -> Rule:
        rule = self.__rule_from_example(sample)
        return self.__generalise(rule, sample)

    def __create_ruleset(self, dataset: pd.DataFrame) -> IndexedRuleSet:
        ruleset = IndexedRuleSet.create_indexed_ruleset(dataset)
        for index, sample in dataset.iloc[:, :-1].iterrows():
            prediction = list(self.predictor.predict(sample.to_frame().transpose()))[0]
            rules = ruleset.get(self.__output_mapping[prediction])
            if not self.__covers(sample, rules):
                rules.append(self.__create_new_rule(sample))
        return ruleset.optimize()

    def __create_theory(self, dataset: pd.DataFrame, ruleset: IndexedRuleSet) -> MutableTheory:
        theory = mutable_theory()
        for key, rule in ruleset.flatten():
            variables = create_variable_list(self.discretization)
            theory.assertZ(self.__create_clause(dataset, variables, key, rule))
        return theory

    def __generalise(self, rule: Rule, sample: pd.Series) -> Rule:
        mutable_rule = rule.to_lists()
        samples = sample.to_frame().transpose()
        for predicates, mutable_predicates in zip(rule.to_lists(), mutable_rule):
            for predicate in predicates:
                samples = self.__remove_antecedents(samples, predicate, mutable_predicates)
        return Rule(mutable_rule[0], mutable_rule[1])

    @lru_cache(maxsize=512)
    def __get_or_set(self, dataset: HashableDataFrame) -> IndexedRuleSet:
        return self.__create_ruleset(dataset)

    def __predict(self, sample: pd.Series):
        sample = get_discrete_dataset(sample.to_frame().transpose(), self.discretization).loc[0]
        x = [index for index, rule in self.__ruleset.flatten() if self.__rule_from_example(sample).is_sub_rule_of(rule)]
        reverse_mapping = dict((v, k) for k, v in self.__output_mapping.items())
        return reverse_mapping[x[0]] if len(x) > 0 else -1

    def __remove_antecedents(self, samples: pd.DataFrame, predicate: str,
                             mutable_predicates: list[str]) -> pd.DataFrame:
        data, is_subset = self.__subset(samples, predicate)
        if is_subset:
            mutable_predicates.remove(predicate)
            return data
        return samples

    def __rule_from_example(self, sample: pd.Series) -> Rule:
        true_predicates, false_predicates = [], []
        for feature, value in sample.items():
            true_predicates.append(str(feature)) if value == 1 else false_predicates.append(str(feature))
        return Rule(sorted(true_predicates), sorted(false_predicates)).reduce(self.discretization)

    def __subset(self, samples: pd.DataFrame, predicate: str) -> (pd.DataFrame, bool):
        samples_0 = samples.copy()
        samples_0[predicate].values[:] = 0
        samples_1 = samples.copy()
        samples_1[predicate].values[:] = 1
        samples_all = samples_0.append(samples_1)
        return samples_all, len(set(self.predictor.predict(samples_all))) == 1

    def extract(self, dataset: pd.DataFrame) -> Theory:
        # Order the dataset by column to preserve reproducibility.
        dataset = dataset.sort_values(by=list(dataset.columns.values), ascending=False)
        # Always perform output mapping in the same (sorted) way to preserve reproducibility.
        self.__output_mapping = {value: index for index, value in enumerate(sorted(set(dataset.iloc[:, -1])))}
        self.__ruleset = self.__get_or_set(HashableDataFrame(dataset))
        return self.__create_theory(dataset, self.__ruleset)

    def predict(self, dataset) -> list:
        return [self.__predict(data.transpose()) for _, data in dataset.iterrows()]