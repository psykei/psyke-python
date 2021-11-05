import math


class Value:

    def __init__(self):
        pass


class Interval(Value):

    def __init__(self, lower: float, upper: float):
        super().__init__()
        self.lower = lower
        self.upper = upper


class LessThan(Interval):

    def __init__(self, value: float):
        super().__init__(- math.inf, value)
        self.value = value
        

class GreaterThan(Interval):

    def __init__(self, value: float):
        super().__init__(value, math.inf)
        self.value = value
        

class Between(Interval):
    
    def __init__(self, lowerbound: float, upperbound: float):
        super().__init__(lowerbound, upperbound)


class Constant(Value):

    def __init__(self, value):
        super().__init__()
        self.value = value
