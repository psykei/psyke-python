from setuptools import setup, find_packages
import pathlib
import subprocess
import distutils.cmd

# current directory
here = pathlib.Path(__file__).parent.resolve()

version_file = here / 'VERSION'

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')


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
        from test.psyke import Predictor
        from psyke.utils import get_default_random_seed
        from psyke.utils.dataframe import get_discrete_dataset
        from sklearn.model_selection import train_test_split
        from test import REQUIRED_PREDICTORS, get_dataset, get_model, get_schema
        from test.resources.predictors import get_predictor_path, PATH, create_predictor_name
        import ast
        import pandas as pd

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
                    schema = get_schema(dataset, int(row['bins']))
                    dataset = get_discrete_dataset(dataset.iloc[:, :-1], schema).join(dataset.iloc[:, -1])
                model = get_model(row['model'], options)
                training_set, test_set = train_test_split(dataset, test_size=0.5,
                                                          random_state=get_default_random_seed())
                model.fit(training_set.iloc[:, :-1], training_set.iloc[:, -1])
                predictor = Predictor(model)
                predictor.save_to_onnx(PATH / file_name, Predictor.get_initial_types(training_set.iloc[:, :-1]))

        required_predictors.to_csv(REQUIRED_PREDICTORS, sep=';', index=False)

        print("Done")


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
    packages=find_packages(),  # Required
    include_package_data=True,
    python_requires='>=3.9.0, <3.10',
    install_requires=[
        'numpy~=1.23.0',
        'pandas~=1.4.3',
        'scikit-learn~=1.1.1',
        '2ppy>=0.3.3',
        # 'skl2onnx~=1.10.0',
        # 'onnxruntime~=1.9.0'
    ],  # Optional
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
    },
)