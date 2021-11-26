from __future__ import annotations
import math


class DiscreteFeature:

    def __init__(self, name: str, admissible_values: dict[str, Value]):
        self.name = name
        self.admissible_values = admissible_values


class Value:

    def __init__(self):
        pass

    def is_in(self, other_value: float) -> bool:
        return False


class Interval(Value):

    def __init__(self, lower: float, upper: float):
        super().__init__()
        self.lower = lower
        self.upper = upper


class LessThan(Interval):

    def __init__(self, value: float):
        super().__init__(- math.inf, value)
        self.value = value

    def is_in(self, other_value: float) -> bool:
        return other_value <= self.value


class GreaterThan(Interval):

    def __init__(self, value: float):
        super().__init__(value, math.inf)
        self.value = value

    def is_in(self, other_value: float) -> bool:
        return other_value > self.value


class Between(Interval):

    def __init__(self, lowerbound: float, upperbound: float):
        super().__init__(lowerbound, upperbound)

    def is_in(self, other_value: float) -> bool:
        return self.lower < other_value <= self.upper


class Constant(Value):

    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def is_in(self, other_value: float) -> bool:
        return math.isclose(other_value, self.value)
