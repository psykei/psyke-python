import ast
import csv
from io import StringIO
import numpy as np
from psyke.predictor import Predictor
from psyke.utils import get_default_random_seed, ONNX_EXTENSION
from sklearn.model_selection import train_test_split
from test import REQUIRED_PREDICTORS, get_dataset, get_model, CLASSPATH
import base64
import os
import pandas as pd
import requests

REPO_URL: str = 'https://api.github.com/repos/psykei/psyke-pytest/contents/resources/'
PL_URL: str = REPO_URL + 'predictors_library.csv'
SEPARATOR: str = ';'

"""
Read the available predictors on psyke-pytest:
hash | file
"""
req = requests.get(PL_URL)
if req.status_code == requests.codes.ok:
    req = req.json()
    content = base64.b64decode(req['content']).decode('utf-8')
else:
    raise FileNotFoundError('Predictors library not found.')
available_predictors = pd.read_csv(StringIO(content), sep=SEPARATOR)

"""
Read the required predictors to run the tests:
hash | type | model | model_options | dataset
"""
required_predictors = pd.read_csv(REQUIRED_PREDICTORS, sep=SEPARATOR)

"""
Retrieve already existing predictors.
"""
for _, row in required_predictors.iterrows():
    if row['hash'] != '' and row['hash'] in available_predictors['hash']:
        predictor_file = available_predictors.loc[available_predictors['hash'] == row['hash']][0]['file']
        req_predictor = requests.get(REPO_URL + predictor_file)
        if req_predictor.status_code == requests.codes.ok:
            req_predictor = req_predictor.json()
            content = base64.b64decode(req['content']).decode('utf-8')
        else:
            raise FileNotFoundError('File ' + predictor_file + ' not found.')
        with open(CLASSPATH + os.path.sep + predictor_file, 'w') as file:
            file.write(content)

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
        predictor.save_to_onnx(CLASSPATH + os.path.sep + file_name,
                               Predictor.get_initial_types(training_set.iloc[:, :-1]))
        with open(CLASSPATH + os.path.sep + 'houseRFR.onnx', 'rb') as file:
            file_content = file.read()
        new_hash = str(hash(file_content))
        required_predictors.loc[index, 'hash'] = new_hash
        new_predictors['hash'].append(new_hash)
        new_predictors['file'].append(file_name)

"""
Update required predictor hash for the new predictors.
"""
required_predictors.to_csv(REQUIRED_PREDICTORS, sep=SEPARATOR, index=False)

"""
Write new predictors in local file, then with github action push it in the pytest repository.
"""
with open(CLASSPATH + os.path.sep + 'new_predictors.csv', 'w') as f:
    w = csv.DictWriter(f, list(new_predictors.keys()))
    w.writeheader()
    w.writerow(new_predictors)
