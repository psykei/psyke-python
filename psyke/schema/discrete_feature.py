from psyke.schema.value import Value


class DiscreteFeature:

    def __init__(self, feature_name: str, admissible_values: dict[str, Value]):
        self.feature_name = feature_name
        self.admissible_values = admissible_values
