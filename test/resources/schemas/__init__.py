from pathlib import Path

from psyke.schema.discrete_feature import DiscreteFeature
from test.resources.schemas.iris import iris_features

PATH = Path(__file__).parents[0]

SCHEMAS: dict[str, DiscreteFeature] = {'iris': iris_features}


def get_schema_path(filename: str) -> Path:
    return PATH / f"{filename}.txt"

