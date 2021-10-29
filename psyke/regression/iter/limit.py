class Limit:

    def __init__(self, feature: str, direction: str):
        self.feature = feature
        self.direction = direction

    def __eq__(self, other):
        return (self.feature == other.feature) & (self.direction == other.direction)

    def __hash__(self):
        return hash(self.feature + self.direction)
