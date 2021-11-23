from typing import Any
from psyke.schema.value import Value

LeafConstraints = list[tuple[str, Value]]
LeafSequence = list[tuple[LeafConstraints, Any]]
