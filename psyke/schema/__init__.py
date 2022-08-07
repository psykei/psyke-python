from __future__ import annotations
import math
from typing import Callable
from psyke.utils import get_int_precision

_EMPTY_INTERSECTION_EXCEPTION: Callable = lambda x, y: \
    Exception("Empty intersection between two Value: " + str(x) + ' and ' + str(y))

_NOT_IMPLEMENTED_INTERSECTION: Callable = lambda x, y: \
    Exception("Not implemented intersection between: " + str(x) + ' and ' + str(y))

_INTERSECTION_WITH_WRONG_TYPE: Callable = lambda x, y: \
    Exception("Calling method with wrong type argument: " + str(x) + ' and ' + str(y))

PRECISION = get_int_precision()
STRING_PRECISION = str(PRECISION)


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
        """
        Check if a real number is inside an interval, or it is equal to a constant.

        :param other: the value to check
        :return: true if the value is inside the interval, false otherwise
        """
        return False

    def is_boundary(self, other: float) -> bool:
        """
        Check if a real number is one edge of an interval, or it is equal to a constant.

        :param other: the value to check
        :return: true if the value is one edge the interval, false otherwise
        """
        if isinstance(self, Constant):
            return self.value == other
        elif isinstance(self, Interval):
            return self.lower == other or self.upper == other
        else:
            return False

    def is_in_or_is_boundary(self, other_value: float) -> bool:
        """
        Check if a real number is in or is boundary for an interval, or for a constant.

        :param other_value: the value to check
        :return: true if at least one condition is true, false otherwise
        """
        return self.is_in(other_value) or self.is_boundary(other_value)

    def __contains__(self, other: Value) -> bool:
        """
        Check if a Value (interval or constant) is contained inside another Value.

        :param other: the other Value
        :return: true if the interval or constant is inside the other interval or constant.
        """
        if isinstance(other, Constant):
            return self.is_in(other.value)
        elif isinstance(other, Interval):
            if type(self) is type(other):
                return self.is_in_or_is_boundary(other.lower) and self.is_in_or_is_boundary(other.upper)
            else:
                return self.is_in(other.lower) and self.is_in(other.upper)
        else:
            return False

    # TODO: handle convention (low priority).
    def __mul__(self, other) -> Value:

        def intersection_with_constant(first_value: Constant, second_value: Value) -> Value:
            if isinstance(first_value, Constant):
                if second_value.is_in(first_value.value):
                    return first_value
                else:
                    raise _EMPTY_INTERSECTION_EXCEPTION(first_value, second_value)
            else:
                raise _INTERSECTION_WITH_WRONG_TYPE(first_value, second_value)

        def intersection_with_outside(first_value: Outside, second_value: Value) -> Value:
            if isinstance(first_value, Outside):
                if isinstance(second_value, LessThan):
                    if second_value.value <= first_value.lower:
                        return second_value
                    elif first_value.is_in(second_value.value):
                        return LessThan(first_value.lower)
                    else:
                        raise _NOT_IMPLEMENTED_INTERSECTION(first_value, second_value)
                elif isinstance(second_value, GreaterThan):
                    if second_value.value >= first_value.lower:
                        return GreaterThan(first_value.upper)
                    elif first_value.is_in(second_value.value):
                        return second_value
                    else:
                        raise _NOT_IMPLEMENTED_INTERSECTION(first_value, second_value)
                elif isinstance(second_value, Constant):
                    if not first_value.is_in(second_value.value):
                        return second_value
                    else:
                        raise _EMPTY_INTERSECTION_EXCEPTION(first_value, second_value)
                elif isinstance(second_value, Between):
                    if second_value in first_value:
                        return second_value
                    elif second_value.lower <= first_value.lower <= second_value.upper <= first_value.upper:
                        return Between(second_value.lower, first_value.lower)
                    elif first_value.lower <= second_value.lower <= first_value.upper <= second_value.upper:
                        return Between(first_value.upper, second_value.upper)
                    else:
                        raise _NOT_IMPLEMENTED_INTERSECTION(first_value, second_value)
                elif isinstance(second_value, Outside):
                    if second_value.lower <= first_value.lower and second_value.upper >= first_value.upper:
                        return second_value
                    elif first_value.lower <= second_value.lower and first_value.upper >= second_value.upper:
                        return first_value
                    else:
                        raise _EMPTY_INTERSECTION_EXCEPTION(first_value, second_value)
                elif isinstance(second_value, Constant):
                    intersection_with_constant(second_value, first_value)
                else:
                    raise _INTERSECTION_WITH_WRONG_TYPE(first_value, second_value)
            else:
                raise _INTERSECTION_WITH_WRONG_TYPE(first_value, second_value)

        def intersection_with_between(first_value: Between, second_value: Value) -> Value:
            if isinstance(first_value, Between):
                if isinstance(second_value, LessThan):
                    if second_value.value <= first_value.lower:
                        raise _EMPTY_INTERSECTION_EXCEPTION(first_value, second_value)
                    elif first_value.lower <= second_value.value <= first_value.upper:
                        return Between(first_value.lower, second_value.value)
                    else:
                        return first_value
                elif isinstance(second_value, GreaterThan):
                    if second_value.value <= first_value.lower:
                        return first_value
                    elif first_value.lower <= second_value.value <= first_value.upper:
                        return Between(second_value.value, first_value.upper)
                    else:
                        raise _EMPTY_INTERSECTION_EXCEPTION(first_value, second_value)
                elif isinstance(second_value, Between):
                    if second_value in first_value:
                        return second_value
                    elif first_value in second_value:
                        return first_value
                    elif first_value.lower <= second_value.lower <= first_value.upper:
                        return Between(second_value.lower, first_value.upper)
                    elif second_value.lower <= first_value.lower <= second_value.upper:
                        return Between(first_value.lower, second_value.upper)
                    elif first_value.lower <= second_value.upper <= first_value.upper:
                        return Between(first_value.lower, second_value.upper)
                    elif second_value.lower <= first_value.upper <= second_value.upper:
                        return Between(second_value.lower, first_value.upper)
                    else:
                        raise _EMPTY_INTERSECTION_EXCEPTION(first_value, second_value)
                elif isinstance(second_value, Constant):
                    intersection_with_constant(second_value, first_value)
                elif isinstance(second_value, Outside):
                    return intersection_with_outside(second_value, first_value)
                else:
                    raise _INTERSECTION_WITH_WRONG_TYPE(first_value, second_value)
            else:
                raise _INTERSECTION_WITH_WRONG_TYPE(first_value, second_value)

        def intersection_with_less_than(first_value: LessThan, second_value: Value) -> Value:
            if isinstance(first_value, LessThan):
                if isinstance(second_value, LessThan):
                    return first_value if first_value in second_value else second_value
                elif isinstance(second_value, GreaterThan):
                    if second_value.value <= first_value.value:
                        return Between(second_value.value, first_value.value)
                    else:
                        raise _EMPTY_INTERSECTION_EXCEPTION(first_value, second_value)
                elif isinstance(second_value, Constant):
                    return intersection_with_constant(second_value, first_value)
                elif isinstance(second_value, Outside):
                    return intersection_with_outside(second_value, first_value)
                elif isinstance(second_value, Between):
                    return intersection_with_between(second_value, first_value)
                else:
                    raise _INTERSECTION_WITH_WRONG_TYPE(first_value, second_value)
            else:
                raise _INTERSECTION_WITH_WRONG_TYPE(first_value, second_value)

        def intersection_with_greater_than(first_value: GreaterThan, second_value: Value) -> Value:
            if isinstance(first_value, GreaterThan):
                if isinstance(second_value, GreaterThan):
                    return first_value if first_value in second_value else second_value
                elif isinstance(second_value, Constant):
                    return intersection_with_constant(second_value, first_value)
                elif isinstance(second_value, Outside):
                    return intersection_with_outside(second_value, first_value)
                elif isinstance(second_value, Between):
                    return intersection_with_between(second_value, first_value)
                elif isinstance(second_value, LessThan):
                    return intersection_with_less_than(second_value, first_value)
                else:
                    raise _INTERSECTION_WITH_WRONG_TYPE(first_value, second_value)
            else:
                raise _INTERSECTION_WITH_WRONG_TYPE(first_value, second_value)

        if other is None:
            return self
        elif isinstance(self, Constant):
            return intersection_with_constant(self, other)
        elif isinstance(self, Outside):
            return intersection_with_outside(self, other)
        elif isinstance(self, Between):
            return intersection_with_between(self, other)
        elif isinstance(self, LessThan):
            return intersection_with_less_than(self, other)
        elif isinstance(self, GreaterThan):
            return intersection_with_greater_than(self, other)
        else:
            raise _INTERSECTION_WITH_WRONG_TYPE(self, other)


class Interval(Value):

    def __init__(self, lower: float, upper: float, standard: bool = True):
        super().__init__()
        self.standard = standard
        self.lower = round(lower, PRECISION)
        self.upper = round(upper, PRECISION)

    def __str__(self):
        return f"[{self.lower:.2f}, {self.upper:.2f}]"

    def __repr__(self):
        return f"Interval({self.lower:.2f}, {self.upper:.2f})"

    def __eq__(self, other: Between) -> bool:
        return (self.upper == other.upper) and (self.lower == other.lower) and (self.standard == other.standard)


class LessThan(Interval):

    def __init__(self, value: float, standard: bool = True):
        super().__init__(-math.inf, value, standard)

    def is_in(self, other: float) -> bool:
        return other <= self.upper if self.standard else other < self.upper

    @property
    def value(self) -> float:
        return self.upper

    def __str__(self):
        return f"]-∞, {self.upper:.2f}" + ("]" if self.standard else "[")

    def __repr__(self):
        return f"LessThan({self.upper:.2f})"

    def __eq__(self, other: LessThan) -> bool:
        return (self.upper == other.upper) and (self.value == other.value) and (self.standard == other.standard)


class GreaterThan(Interval):

    def __init__(self, value: float, standard: bool = True):
        super().__init__(value, math.inf, standard)

    def is_in(self, other: float) -> bool:
        return other > self.lower if self.standard else other >= self.lower

    @property
    def value(self) -> float:
        return self.lower

    def __str__(self):
        return ("]" if self.standard else "[") + f"{self.lower:.2f}, ∞["

    def __repr__(self):
        return f"GreaterThan({self.lower:.2f})"

    def __eq__(self, other: GreaterThan) -> bool:
        return (self.lower == other.lower) and (self.value == other.value) and (self.standard == other.standard)


class Between(Interval):

    def __init__(self, lowerbound: float, upperbound: float, standard: bool = True):
        super().__init__(lowerbound, upperbound, standard)

    def is_in(self, other: float) -> bool:
        return self.lower <= other < self.upper if self.standard else self.lower < other <= self.upper

    def __str__(self):
        return ("[" if self.standard else "]") + f"{self.lower:.2f}, {self.upper:.2f}" + ("[" if self.standard else "]")

    def __repr__(self):
        return f"Between({self.lower:.2f}, {self.upper:.2f})"


class Outside(Interval):

    def __init__(self, lowerbound: float, upperbound: float, standard: bool = True):
        super().__init__(lowerbound, upperbound, standard)

    def is_in(self, other: float) -> bool:
        return other < self.lower or self.upper <= other if self.standard else other <= self.lower or self.upper < other

    def __str__(self):
        return f"]-∞, {self.lower:.2f}" + ("[" if self.standard else "]") + ' U '\
               + ("[" if self.standard else "]") + f"{self.upper:.2f}, ∞["

    def __repr__(self):
        return f"Outside({self.lower:.2f}, {self.upper:.2f})"


class Constant(Value):

    def __init__(self, value: float):
        super().__init__()
        self.value = round(value, get_int_precision())

    def is_in(self, other: float) -> bool:
        return math.isclose(other, self.value)

    def __str__(self):
        return "{" + str(self.value) + "}"

    def __repr__(self):
        return f"Constant({self.value})"


def term_to_value(term) -> Value:

    real_to_float: Callable = lambda x: float(str(x)) if not hasattr(x, 'len') else float(str(x[0]))

    functor_to_value = {
        '<': lambda x: LessThan(real_to_float(x), standard=False),
        '=<': lambda x: LessThan(real_to_float(x)),
        '>': lambda x: GreaterThan(real_to_float(x)),
        '>=': lambda x: GreaterThan(real_to_float(x), standard=False),
        '==': lambda x: Constant(real_to_float(x)),
        'in': lambda x: Between(real_to_float(x[0]), real_to_float(x[1][0])),
        'not_in': lambda x: Outside(real_to_float(x[0]), real_to_float(x[1][0]))
    }
    return functor_to_value[term.functor](term.args[1])
