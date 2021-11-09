from __future__ import annotations
from onnxconverter_common import DataType, Int64TensorType, FloatTensorType, StringTensorType
from skl2onnx import convert_sklearn
from typing import Iterable
import numpy as np
import onnxruntime as rt
import pandas as pd


class Predictor:

    def __init__(self, model, from_file_onnx=False):
        self._model = model
        self._from_file_onnx = from_file_onnx

    @staticmethod
    def load_from_onnx(file: str) -> Predictor:
        return Predictor(rt.InferenceSession(file), True)

    def save_to_onnx(self, file, initial_types: list[tuple[str, DataType]]):
        if not self._from_file_onnx:
            onnx_predictor = convert_sklearn(self._model, initial_types=initial_types)
            with open(file, 'wb') as f:
                f.write(onnx_predictor.SerializeToString())

    def predict(self, dataset: pd.DataFrame) -> Iterable:
        if self._from_file_onnx:
            input_name = self._model.get_inputs()[0].name
            label_name = self._model.get_outputs()[0].name
            pred_onx = self._model.run([label_name], {input_name: dataset.to_numpy().astype(np.float32)})[0]
            return [prediction for plist in pred_onx for prediction in plist]
        else:
            return self._model.predict(dataset)

    @staticmethod
    def get_initial_types(dataset: pd.DataFrame) -> list[tuple[str, DataType]]:
        array = dataset.to_numpy()
        name = ''
        for column in dataset.columns:
            name += column + ', '
        name = name[:-2]
        shape = [None, array.shape[1]]
        if array.dtype == 'int64':
            types = Int64TensorType(shape)
        elif array.dtype == 'float64':
            types = FloatTensorType(shape)
        else:
            types = StringTensorType(shape)
        return [(name, types)]
