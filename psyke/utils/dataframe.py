from hashlib import sha256
from typing import Iterable, List
import pandas as pd
from pandas.core.util.hashing import hash_pandas_object

from psyke import DiscreteFeature
from psyke.schema import LessThan, GreaterThan, Between, Value


def split_features(dataframe: pd.DataFrame) -> Iterable[DiscreteFeature]:
    result = []
    features = {'V' + str(index + 1): column for index, column in enumerate(dataframe.columns)}
    for feature, column in features.items():
        values = set(dataframe[column])
        result.append(DiscreteFeature(feature, {feature + '_' + str(i): v for i, v in enumerate(values)}))
    return result


def get_discrete_features_equal_frequency(
        dataframe: pd.DataFrame,
        bins: int = None,
        output=True,
        bin_names: List[str] = []
) -> Iterable[DiscreteFeature]:
    features = dataframe.columns[:-1] if output else dataframe.columns
    result = set()
    if bins is None:
        if len(bin_names) > 0:
            bins = len(bin_names)
        else:
            raise ValueError("No bins nor bin_names have been provided")
    elif bins > 0:
        if len(bin_names) == 0:
            bin_names = range(0, bins)
        elif len(bin_names) == bins:
            pass
        else:
            raise ValueError("Mismatch among the provided amount of bins and the bin_names")
    else:
        raise ValueError("Negative amount of bins makes no sense")
    for feature in features:
        values = sorted(dataframe[feature])
        intervals = [values[i] for i in range(int(len(values) / bins), len(values), int(len(values) / bins))]
        starting_interval: list[Value] = [LessThan(intervals[0])]
        ending_interval: list[Value] = [GreaterThan(intervals[-1])]
        middle_intervals: list[Value] = [Between(intervals[i], intervals[i + 1]) for i in range(0, len(intervals) - 1)]
        new_intervals = starting_interval + middle_intervals + ending_interval
        new_feature_names = [feature + '_' + str(i) for i in range(0, bins)]
        new_features = {new_feature_names[i]: new_intervals[i] for i in range(0, bins)}
        result.add(DiscreteFeature(feature, new_features))
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


class HashableDataFrame(pd.DataFrame):
    def __init__(self, obj):
        super().__init__(obj)

    def __hash__(self):
        hash_value = sha256(hash_pandas_object(self, index=True).values)
        hash_value = hash(hash_value.hexdigest())
        return hash_value

    def __eq__(self, other):
        return self.equals(other)