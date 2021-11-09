from psyke.schema.discrete_feature import DiscreteFeature
from psyke.schema.value import LessThan, Between, GreaterThan


iris_features = {
    DiscreteFeature(
        "SepalLength",
        {
            "SepalLength_0": LessThan(5.39),
            "SepalLength_1": Between(5.39, 6.26),
            "SepalLength_2": GreaterThan(6.26)
        }),
    DiscreteFeature(
        "SepalWidth",
        {
            "SepalWidth_0": LessThan(2.87),
            "SepalWidth_1": Between(2.87, 3.2),
            "SepalWidth_2": GreaterThan(3.2)
        }),
    DiscreteFeature(
        "PetalLength",
        {
            "PetalLength_0": LessThan(2.28),
            "PetalLength_1": Between(2.28, 4.87),
            "PetalLength_2": GreaterThan(4.87)
        }),
    DiscreteFeature(
        "PetalWidth",
        {
            "PetalWidth_0": LessThan(0.65),
            "PetalWidth_1": Between(0.65, 1.64),
            "PetalWidth_2": GreaterThan(1.64)
        })
}
