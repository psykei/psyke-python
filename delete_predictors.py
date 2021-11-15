import os

from psyke.utils import ONNX_EXTENSION
from test import CLASSPATH

files_in_directory = os.listdir(CLASSPATH)
filtered_files = [file for file in files_in_directory if file.endswith(ONNX_EXTENSION)]
for file in filtered_files:
    path_to_file = os.path.join(CLASSPATH, file)
    os.remove(path_to_file)
