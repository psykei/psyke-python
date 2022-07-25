from __future__ import annotations
import math
from typing import Callable


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

    def is_in(self, other: float) -> bool:
        return False

    def is_boundary(self, other: float) -> bool:
        if isinstance(self, Constant):
            return self.value == other
        elif isinstance(self, Interval):
            return self.lower == other or self.upper == other
        else:
            return False

    def is_in_or_is_boundary(self, other_value: float) -> bool:
        return self.is_in(other_value) or self.is_boundary(other_value)

    def __contains__(self, other: Value) -> bool:
        if isinstance(other, Constant):
            return self.is_in(other.value)
        elif isinstance(other, Interval):
            if type(self) is type(other):
                return self.is_in_or_is_boundary(other.lower) and self.is_in_or_is_boundary(other.upper)
            else:
                return self.is_in(other.lower) and self.is_in(other.upper)
        else:
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

    def is_in(self, other: float) -> bool:
        return other <= self.value

    def __str__(self):
        return f"]-∞, {self.value:.2f}["

    def __repr__(self):
        return f"LessThan({self.value:.2f})"


class GreaterThan(Interval):

    def __init__(self, value: float):
        super().__init__(value, math.inf)
        self.value = value

    def is_in(self, other: float) -> bool:
        return other > self.value

    def __str__(self):
        return f"]{self.value:.2f}, ∞["

    def __repr__(self):
        return f"GreaterThan({self.value:.2f})"


class Between(Interval):

    def __init__(self, lowerbound: float, upperbound: float):
        super().__init__(lowerbound, upperbound)

    def is_in(self, other: float) -> bool:
        return self.lower < other <= self.upper

    def __repr__(self):
        return f"Between({self.lower:.2f}, {self.upper:.2f})"


class Constant(Value):

    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def is_in(self, other: float) -> bool:
        return math.isclose(other, self.value)

    def __str__(self):
        return "{" + str(self.value) + "}"

    def __repr__(self):
        return f"Constant({self.value})"


def term_to_value(term) -> Value:

    real_to_float: Callable = lambda x: float(str(x)) if not hasattr(x, 'len') else float(str(x[0]))

    functor_to_value = {
        '<': lambda x: LessThan(real_to_float(x)),
        '=<': lambda x: LessThan(real_to_float(x)),
        '>': lambda x: GreaterThan(real_to_float(x)),
        '>=': lambda x: GreaterThan(real_to_float(x)),
        '==': lambda x: Constant(real_to_float(x)),
        'in': lambda x: Between(real_to_float(x[0]), real_to_float(x[1][0])),
        'not_in': lambda x: Between(real_to_float(x[0]), real_to_float(x[1][0]))
    }
    return functor_to_value[term.functor](term.args[1])
