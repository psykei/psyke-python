from __future__ import annotations
import math


class DiscreteFeature:

    def __init__(self, name: str, admissible_values: dict[str, Value]):
        self.name = name
        self.admissible_values = admissible_values

    def __str__(self):
        return self.name + " = {" \
            + ", ".join([f"'{k}' if {self.name} ∈ {str(self.admissible_values[k])}" for k in self.admissible_values]) \
            + "}"

    def __repr__(self):
        return f"DiscreteFeature(name={self.name}, admissible_values={self.admissible_values})"


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

    def __str__(self):
        return f"[{self.lower:.2f}, {self.upper:.2f}]"

    def __repr__(self):
        return f"Interval({self.lower:.2f}, {self.upper:.2f})"


class LessThan(Interval):

    def __init__(self, value: float):
        super().__init__(-math.inf, value)
        self.value = value

    def is_in(self, other_value: float) -> bool:
        return other_value <= self.value

    def __str__(self):
        return f"]-∞, {self.value:.2f}["

    def __repr__(self):
        return f"LessThan({self.value:.2f})"


class GreaterThan(Interval):

    def __init__(self, value: float):
        super().__init__(value, math.inf)
        self.value = value

    def is_in(self, other_value: float) -> bool:
        return other_value > self.value

    def __str__(self):
        return f"]{self.value:.2f}, ∞["

    def __repr__(self):
        return f"GreaterThan({self.value:.2f})"


class Between(Interval):

    def __init__(self, lowerbound: float, upperbound: float):
        super().__init__(lowerbound, upperbound)

    def is_in(self, other_value: float) -> bool:
        return self.lower < other_value <= self.upper

    def __repr__(self):
        return f"Between({self.lower:.2f}, {self.upper:.2f})"


class Constant(Value):

    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def is_in(self, other_value: float) -> bool:
        return math.isclose(other_value, self.value)

    def __str__(self):
        return "{" + str(self.value) + "}"

    def __repr__(self):
        return f"Constant({self.value})"