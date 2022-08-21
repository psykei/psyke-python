from functools import lru_cache
from psyke.extraction.real.utils import Rule, IndexedRuleSet
from psyke import Extractor
from psyke.schema import DiscreteFeature
from psyke.utils.dataframe import HashableDataFrame
from psyke.utils.logic import create_term, create_head, create_variable_list
from tuprolog.core import Var, Struct, Clause, clause
from tuprolog.theory import MutableTheory, mutable_theory, Theory
from typing import Iterable
import pandas as pd
import numpy as np


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

    @property
    def n_rules(self):
        return len(self.__ruleset.flatten())

    def __covers(self, sample: pd.Series, rules: list[Rule]) -> bool:
        new_rule = self.__rule_from_example(sample)
        return any([new_rule in rule for rule in rules])

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
        for predicate in rule.true_predicates:
            samples = self.__remove_antecedent(samples.copy(), predicate, mutable_rule)
        return Rule(mutable_rule[0], mutable_rule[1]).reduce(self.discretization)

    def __remove_antecedent(self, samples: pd.DataFrame, predicate: str, rule: list[list[str]]) -> (pd.DataFrame, bool):
        feature = [feature for feature in self.discretization if predicate in feature.admissible_values][0]
        output = np.array(self.predictor.predict(samples))
        copies = [samples.copy()]
        samples[predicate] = 0
        for f in [f for f in feature.admissible_values if f != predicate]:
            copy = samples.copy()
            copy[f] = 1
            if all(output == np.array(self.predictor.predict(copy))):
                copies.append(copy)
                rule[1].remove(f)
        if len(copies) > 1:
            rule[0].remove(predicate)
        return pd.concat([df for df in copies], ignore_index=True)

    @lru_cache(maxsize=512)
    def __get_or_set(self, dataset: HashableDataFrame) -> IndexedRuleSet:
        return self.__create_ruleset(dataset)

    def __predict(self, sample: pd.Series):
        x = [index for index, rule in self.__ruleset.flatten() if self.__rule_from_example(sample) in rule]
        reverse_mapping = dict((v, k) for k, v in self.__output_mapping.items())
        return reverse_mapping[x[0]] if len(x) > 0 else None

    def __rule_from_example(self, sample: pd.Series) -> Rule:
        true_predicates, false_predicates = [], []
        for feature, value in sample.items():
            true_predicates.append(str(feature)) if value == 1 else false_predicates.append(str(feature))
        return Rule(sorted(true_predicates), sorted(false_predicates))

    def __subset(self, samples: pd.DataFrame, predicate: str) -> (pd.DataFrame, bool):
        samples_0 = samples.copy()
        samples_0[predicate].values[:] = 0
        samples_1 = samples.copy()
        samples_1[predicate].values[:] = 1
        samples_all = samples_0.append(samples_1)
        return samples_all, len(set(self.predictor.predict(samples_all))) == 1

    def extract(self, dataframe: pd.DataFrame) -> Theory:
        # Order the dataset by column to preserve reproducibility.
        dataframe = dataframe.sort_values(by=list(dataframe.columns.values), ascending=False)
        # Always perform output mapping in the same (sorted) way to preserve reproducibility.
        self.__output_mapping = {value: index for index, value in enumerate(sorted(set(dataframe.iloc[:, -1])))}
        self.__ruleset = self.__get_or_set(HashableDataFrame(dataframe))
        return self.__create_theory(dataframe, self.__ruleset)

    def predict(self, dataframe) -> Iterable:
        return np.array([self.__predict(data.transpose()) for _, data in dataframe.iterrows()])
