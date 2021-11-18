from typing import Iterable
import pandas as pd
from psyke.schema.discrete_feature import DiscreteFeature


def split_features(dataframe: pd.DataFrame) -> Iterable[DiscreteFeature]:
    result = []
    features = {'V' + str(index + 1): column for index, column in enumerate(dataframe.columns)}
    for feature, column in features.items():
        values = set(dataframe[column])
        result.append(DiscreteFeature(feature, {feature + '_' + str(i): v for i, v in enumerate(values)}))
    return result


def get_discrete_dataset(dataset: pd.DataFrame, discrete_features: Iterable[DiscreteFeature],
                         sort: bool = True) -> pd.DataFrame:
    """
    Create a new dataset mapping the old features into the new discrete features.
    Note: some algorithms require the same SORTED feature to be replicable due to rule optimization and other stuffs.
    Therefore the new features are alphabetically sorted.
    This is not strictly necessary because internally those algorithms perform the sorting themself.
    However it is a good idea to have this same function returning the same result w.r.t. the inputs.

    :param dataset: the original dataset
    :param discrete_features: mapping for the features
    :param sort: alphabetically sort new features
    :return: the new discrete dataset
    """
    columns_name = [key for feature in discrete_features for key, _ in feature.admissible_values.items()]
    if sort:
        columns_name = sorted(columns_name)
    new_dataset = pd.DataFrame(columns=columns_name)
    for feature in discrete_features:
        for index, value in enumerate(dataset[feature.name]):
            for key, admissible_value in feature.admissible_values.items():
                new_dataset.loc[index, key] = int(admissible_value.is_in(value))

    for feature in discrete_features:
        for new_feature in feature.admissible_values.keys():
            new_dataset[new_feature] = new_dataset[new_feature].astype(str).astype(int)

    return new_dataset
