from __future__ import annotations

from psyke.schema.discrete_feature import DiscreteFeature


class Rule:

    def __init__(self, true_predicates: list[str], false_predicates: list[str]):
        self.true_predicates = true_predicates
        self.false_predicates = false_predicates

    def is_sub_rule_of(self, rule: Rule) -> bool:
        return all([predicate in self.true_predicates for predicate in rule.true_predicates]) & \
               all([predicate in self.false_predicates for predicate in rule.false_predicates])

    def reduce(self, features: set[DiscreteFeature]) -> Rule:
        to_be_removed = ((feature.admissible_values.keys for feature in features if tp in feature.admissible_values)
                         for tp in self.true_predicates)
        return Rule(self.true_predicates, [fp for fp in self.false_predicates if fp not in to_be_removed])

    def to_lists(self) -> list[list[str]]:
        return [self.true_predicates, self.false_predicates]
