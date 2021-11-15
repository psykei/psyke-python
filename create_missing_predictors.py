import ast
from psyke.predictor import Predictor
from psyke.utils import get_default_random_seed, ONNX_EXTENSION
from sklearn.model_selection import train_test_split
from pytest.resources import CLASSPATH as PYTEST_CLASSPATH
from pytest.resources import PL
from pytest.utils import update_library
from test import REQUIRED_PREDICTORS, get_dataset, get_model
import os
import pandas as pd

SEPARATOR: str = ';'

"""
Read the available predictors on pytest:
hash | file
"""
available_predictors = pd.read_csv(PL, sep=SEPARATOR)

"""
Read the required predictors to run the tests:
hash | type | model | model_options | dataset
"""
required_predictors = pd.read_csv(REQUIRED_PREDICTORS, sep=SEPARATOR)

"""
Create missing predictors.
"""
new_predictors: dict[str, list] = {'hash': [], 'file': []}
for index, row in required_predictors.iterrows():
    if row['hash'] == '' or row['hash'] not in available_predictors['hash']:
        dataset = get_dataset(row['dataset'])
        model = get_model(row['model'], ast.literal_eval(row['model_options']))
        training_set, test_set = train_test_split(dataset, test_size=0.5, random_state=get_default_random_seed())
        model.fit(training_set.iloc[:, :-1], training_set.iloc[:, -1])
        file_name = row['dataset'] + row['model'] + ONNX_EXTENSION
        predictor = Predictor(model)
        predictor.save_to_onnx(PYTEST_CLASSPATH + os.path.sep + file_name,
                               Predictor.get_initial_types(training_set.iloc[:, :-1]))
        with open(PYTEST_CLASSPATH + os.path.sep + file_name, 'rb') as file:
            file_content = file.read()
        new_hash = str(hash(file_content))
        required_predictors.loc[index, 'hash'] = new_hash
        new_predictors['hash'].append(new_hash)
        new_predictors['file'].append(file_name)
        update_library(file_name)

"""
Update required predictor hash for the new predictors.
"""
required_predictors.to_csv(REQUIRED_PREDICTORS, sep=SEPARATOR, index=False)


