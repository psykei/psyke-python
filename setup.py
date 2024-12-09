from setuptools import setup, find_packages
import pathlib
import subprocess
import distutils.cmd

here = pathlib.Path(__file__).parent.resolve()

version_file = here / 'VERSION'

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')


EPOCHS: int = 50
BATCH_SIZE: int = 16
REQUIREMENTS = [
    'numpy~=1.26.0',
    'pandas~=2.2.0',
    'scikit-learn~=1.6.0',
    '2ppy~=0.4.0',
    'kneed~=0.8.1',
    'sympy~=1.11'
]  # Optional


def format_git_describe_version(version):
    if '-' in version:
        splitted = version.split('-')
        tag = splitted[0]
        index = f"dev{splitted[1]}"
        return f"{tag}.{index}"
    else:
        return version


def get_version_from_git():
    try:
        process = subprocess.run(["git", "describe"], cwd=str(here), check=True, capture_output=True)
        version = process.stdout.decode('utf-8').strip()
        version = format_git_describe_version(version)
        with version_file.open('w') as f:
            f.write(version)
        return version
    except subprocess.CalledProcessError:
        if version_file.exists():
            return version_file.read_text().strip()
        else:
            return '0.1.0.archeo'


version = get_version_from_git()


print(f"Detected version {version} from git describe")


class GetVersionCommand(distutils.cmd.Command):
    """A custom command to get the current project version inferred from git describe."""

    description = 'gets the project version from git describe'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        print(version)


class CreateTestPredictors(distutils.cmd.Command):
    description = 'gets the project version from git describe'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        from psyke.utils import get_default_random_seed
        from psyke.utils.dataframe import get_discrete_dataset
        from sklearn.model_selection import train_test_split
        from test import REQUIRED_PREDICTORS, get_dataset, get_model, get_schema
        from test.resources.predictors import get_predictor_path, PATH, create_predictor_name
        import ast
        import pandas as pd
        from tensorflow.keras import Model
        from test import Predictor

        # Read the required predictors to run the tests:
        #   model | model_options | dataset
        required_predictors = pd.read_csv(REQUIRED_PREDICTORS, sep=';')

        # Create missing predictors.
        #     model | model_options | dataset
        for index, row in required_predictors.iterrows():
            options = ast.literal_eval(row['model_options'])
            file_name = create_predictor_name(row['dataset'], row['model'], options)
            if not get_predictor_path(file_name).is_file():
                dataset = get_dataset(row['dataset'])
                if row['bins'] > 0:
                    schema = get_schema(dataset)  # int(row['bins'])
                    dataset = get_discrete_dataset(dataset.iloc[:, :-1], schema).join(dataset.iloc[:, -1])
                model, _ = get_model(row['model'], options)
                training_set, test_set = train_test_split(dataset, test_size=0.5,
                                                          random_state=get_default_random_seed())
                if isinstance(model, Model):
                    keys = set(training_set.iloc[:, -1])
                    mapping = {key: i for i, key in enumerate(keys)}
                    training_set.iloc[:, -1] = training_set.iloc[:, -1].apply(lambda x: mapping[x])
                    test_set.iloc[:, -1] = test_set.iloc[:, -1].apply(lambda x: mapping[x])
                    model.fit(training_set.iloc[:, :-1], training_set.iloc[:, -1], epochs=EPOCHS, batch_size=BATCH_SIZE)
                else:
                    model.fit(training_set.iloc[:, :-1], training_set.iloc[:, -1])
                predictor = Predictor(model)
                predictor.save_to_onnx(PATH / file_name, Predictor.get_initial_types(training_set.iloc[:, :-1]))

        required_predictors.to_csv(REQUIRED_PREDICTORS, sep=';', index=False)

        print("Done")


class CreateTheoryPlot(distutils.cmd.Command):
    description = 'create a plot representing samples X and their class/regression value Y predicted by a theory'
    user_options = [('theory=', 't', 'textual file of a Prolog theory'),
                    ('dataset=', 'd', 'file of a dataset'),
                    ('azimuth=', 'a', 'azimuth of the plot'),
                    ('distance=', 'D', 'distance from the plot'),
                    ('elevation=', 'e', 'elevation of the plot'),
                    ('output=', 'o', 'output file name of the plot'),
                    ('show=', 's', 'show theory in the plot ([y]/n)'),
                    ]
    default_output_file_name = 'dummy/plot'
    default_theory_name = 'dummy/iris-theory'
    default_dataset_name = 'dummy/iris'
    default_azimuth = '45'
    default_distance = '9'
    default_elevation = '5'
    csv_format = '.csv'
    txt_format = '.txt'
    pdf_format = '.pdf'

    def initialize_options(self):
        self.output = self.default_output_file_name
        self.theory = self.default_theory_name
        self.dataset = self.default_dataset_name
        self.azimuth = self.default_azimuth
        self.elevation = self.default_elevation
        self.distance = self.default_distance
        self.show = True

    def finalize_options(self):
        self.theory_file = str(self.theory)
        self.data = str(self.dataset)
        self.output = str(self.output)
        self.a = float(self.azimuth)
        self.e = float(self.elevation)
        self.d = float(self.distance)
        self.s = self.show in (True, 'y', 'Y', 'yes', 'YES', 'Yes')

    def run(self):
        import pandas as pd
        from tuprolog.theory.parsing import parse_theory
        from psyke.utils.plot import plot_theory

        if self.theory_file is None or self.theory_file == '':
            raise Exception('Empty theory file name')
        if self.data is None or self.data == '':
            raise Exception('Empty dataset file name')
        with open(self.theory_file + (self.txt_format if '.' not in self.theory_file else ''), 'r') as file:
            textual_theory = file.read()
        theory = parse_theory(textual_theory)
        data = pd.read_csv(self.data + (self.csv_format if '.' not in self.data else ''))
        plot_theory(theory, data, self.output + self.pdf_format, self.a, self.d, self.e, self.s)


setup(
    name='psyke',  # Required
    version=version,
    description='Python-based implementation of PSyKE, i.e. a Platform for Symbolic Knowledge Extraction',
    license='Apache 2.0 License',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/psykei/psyke-python',
    author='Matteo Magnini',
    author_email='matteo.magnini@unibo.it',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Prolog'
    ],
    keywords='knowledge extraction, symbolic ai, ske, extractor, rules, prolog',  # Optional
    # package_dir={'': 'src'},  # Optional
    packages=find_packages('.'),  # Required
    include_package_data=True,
    python_requires='>=3.9.0, <3.10',
    install_requires=REQUIREMENTS,  # Optional
    zip_safe = False,
    platforms = "Independant",
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/psykei/psyke-python/issues',
        # 'Funding': 'https://donate.pypi.org',
        # 'Say Thanks!': 'http://saythanks.io/to/example',
        'Source': 'https://github.com/psykei/psyke-python',
    },
    cmdclass={
        'get_project_version': GetVersionCommand,
        'create_test_predictors': CreateTestPredictors,
        'create_theory_plot': CreateTheoryPlot
    },
)
