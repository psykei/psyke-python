from __future__ import annotations
import pandas as pd

from psyke.classification.real.rule import Rule


class IndexedRuleSet(dict[int, list[Rule]]):

    def flatten(self) -> list[tuple[int, Rule]]:
        return [(key, value) for key, values in self for value in values]

    def optimize(self):
        useless_rules = [IndexedRuleSet.__useless_rules(entry) for entry in self]
        for key, rule in useless_rules:
            self[key].remove(rule)

    @staticmethod
    def __useless_rules(key_rules_or_key, rules=None) -> list[(int, Rule)]:
        if rules is None:
            return IndexedRuleSet.__useless_rules(key_rules_or_key[0], key_rules_or_key[1])
        else:
            return [(key_rules_or_key, rule) for rule in rules
                    if not any(aux.is_sub_rule_of(rule) for aux in [rules - rule])]

    @staticmethod
    def create_indexed_ruleset(dataset: pd.DataFrame) -> IndexedRuleSet:
        return IndexedRuleSet({index: [] for index, _ in enumerate(set(dataset.iloc[:, -1]))})
