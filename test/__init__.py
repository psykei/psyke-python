from __future__ import annotations

import os
from typing import Iterable, Union
import numpy as np
import onnxruntime
import pandas as pd
from tensorflow.python.saved_model.save import save
from onnxconverter_common import FloatTensorType, Int64TensorType, StringTensorType, DataType
from skl2onnx import convert_sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from tensorflow.python.framework.random_seed import set_seed
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from psyke.schema import DiscreteFeature, Value
from psyke.utils import get_default_random_seed
from sklearn.datasets import fetch_california_housing, load_iris
from psyke import Extractor
from psyke.utils.dataframe import get_discrete_features_supervised
from test.resources.predictors import PATH, get_predictor_path

REQUIRED_PREDICTORS: str = PATH / '.required.csv'
LE = '=<'
GE = '>='
L = '<'
G = '>'


def get_extractor(extractor_type: str, parameters: dict):
    if extractor_type.lower() == 'cart':
        return Extractor.cart(**parameters)
    elif extractor_type.lower() == 'iter':
        return Extractor.iter(**parameters)
    elif extractor_type.lower() == 'real':
        return Extractor.real(**parameters)
    elif extractor_type.lower() == 'trepan':
        return Extractor.trepan(**parameters)
    elif extractor_type.lower() == 'gridex':
        return Extractor.gridex(**parameters)
    else:
        raise NotImplementedError(extractor_type + ' not implemented yet.')


def get_model(model_type: str, parameters: dict):
    if model_type.lower() == 'rfr':
        return RandomForestRegressor(**parameters, random_state=np.random.seed(get_default_random_seed())), False
    elif model_type.lower() == 'knnc':
        return KNeighborsClassifier(**parameters), False  # It's deterministic, don't have a random_state
    elif model_type.lower() == 'dtc':
        return DecisionTreeClassifier(max_leaf_nodes=3, random_state=np.random.seed(get_default_random_seed())), False
    elif model_type.lower() == 'dtr':
        return DecisionTreeRegressor(max_depth=3, random_state=np.random.seed(get_default_random_seed())), False
    elif model_type.lower() == 'nn':
        return get_simple_neural_network(**parameters, random_state=np.random.seed(get_default_random_seed())), False
    else:
        return Predictor.load_from_onnx(str(get_predictor_path(model_type))), True


def get_simple_neural_network(input: int = 4, output: int = 3, layers: int = 3, neurons: int = 32,
                              random_state: int = np.random.seed(get_default_random_seed())) -> Model:
    set_seed(random_state)
    input_layer = Input(input)
    x = input_layer
    for _ in range(layers-1):
        x = Dense(neurons, activation='relu')(x)
    x = Dense(output, activation='softmax')(x)
    model = Model(input_layer, x)
    model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def get_dataset(name: str):
    if name.lower() == 'house':
        x, y = fetch_california_housing(return_X_y=True, as_frame=True)
        normalized_x = _normalize_data(x)
        normalized_y = _normalize_data(y)
        return normalized_x.join(normalized_y)
    elif name.lower() == 'iris':
        x, y = load_iris(return_X_y=True, as_frame=True)
        y = pd.DataFrame(y).replace({"target": {0: 'setosa', 1: 'versicolor', 2: 'virginica'}})
        result = x.join(y)
        result.columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'iris']
        return result
    else:
        raise Exception('unknown dataset name.')


def _normalize_data(x: pd.DataFrame) -> pd.DataFrame:
    return (x - x.min()) / (x.max() - x.min())


def get_schema(dataset: pd.DataFrame) -> Union[Iterable[DiscreteFeature], None]:
    return get_discrete_features_supervised(dataset)
    # return SCHEMAS[filename] if filename in SCHEMAS.keys() else None


def _get_admissible_values(prepositions: Iterable[str]) -> dict[str, Value]:
    raise NotImplementedError('Automatic schema reading not implemented yet.')


class Predictor:

    def __init__(self, model, from_file_onnx=False):
        self._model = model
        self._from_file_onnx = from_file_onnx

    @staticmethod
    def load_from_onnx(file: str) -> Predictor:
        return Predictor(onnxruntime.InferenceSession(file), True)

    def save_to_onnx(self, file, initial_types: list[tuple[str, DataType]]):
        file = str(file) + '.onnx'
        if not self._from_file_onnx:
            if os.path.exists(file):
                os.remove(file)
            if isinstance(self._model, Model):
                save(self._model, "tmp_model")
                os.system("python -m tf2onnx.convert --saved-model tmp_model --output " + file)
            else:
                onnx_predictor = convert_sklearn(self._model, initial_types=initial_types)
                with open(file, 'wb') as f:
                    f.write(onnx_predictor.SerializeToString())

    def predict(self, dataset: pd.DataFrame | np.ndarray) -> Iterable:
        array = dataset.to_numpy() if isinstance(dataset, pd.DataFrame) else dataset
        if self._from_file_onnx:
            input_name = self._model.get_inputs()[0].name
            label_name = self._model.get_outputs()[0].name
            if array.dtype == 'float64':
                tensor_type = np.float32
            elif array.dtype == 'int64' or array.dtype == 'int32':
                tensor_type = np.int64
            else:
                tensor_type = np.str
            pred_onx = self._model.run([label_name], {input_name: array.astype(tensor_type)})[0]
            return [prediction for plist in pred_onx for prediction in plist] if isinstance(pred_onx[0], list) \
                else [prediction for prediction in pred_onx]
        else:
            return self._model.predict(dataset)

    # TODO: to be improved, make it more flexible
    @staticmethod
    def get_initial_types(dataset: pd.DataFrame | np.ndarray) -> list[tuple[str, DataType]]:
        array = dataset.to_numpy() if isinstance(dataset, pd.DataFrame) else dataset
        name = ''
        for column in dataset.columns:
            name += column + ', '
        name = name[:-2]
        shape = [None, array.shape[1]]
        if array.dtype == 'float64':
            types = FloatTensorType(shape)
        elif array.dtype == 'int64':
            types = Int64TensorType(shape)
        else:
            types = StringTensorType(shape)
        return [(name, types)]
