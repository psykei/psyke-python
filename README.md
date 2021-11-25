# PSyKE

Some quick links:
* [Home Page](https://apice.unibo.it/xwiki/bin/view/PSyKE/)
* [GitHub Repository](https://github.com/psykei/psyke-python)
* [PyPi Repository](https://pypi.org/project/psyke/)
* [Issues](https://github.com/psykei/psyke-python/issues)

## Intro

[PSyKE](https://apice.unibo.it/xwiki/bin/view/PSyKE/) (Platform for Symbolic Knowledge Extraction)
is intended to be a library that provides support to symbolic knowledge extraction from black-box predictors.

PSyKE offers different algorithms for symbolic knowledge extraction both for classification and regression problems.
The extracted knowledge is a prolog theory (i.e. a set of prolog clauses).
PSyKE relies on [2ppy](https://github.com/tuProlog/2ppy) (tuProlog in Python) for logic support.

Project structure overview:

![PSyKE class diagram](https://github.com/psykei/psyke-python/blob/master/.img/class_diagram.png)

<!--
To generate the class diagram go to this url
//www.plantuml.com/plantuml/png/PLBBRkem4DtdAqQixeLcqsN40aHfLQch2dM341gS0IpoY3oJYfJctnl7QUgwKZRdCUFZ4ozOq4YTPr65we8dWlkgQcuHmEPCfMbW6iDaEe5LLjPCKHj5AaFcGRsr0tHo1ySr5JSNidkwxvlhlVge5Oek2ok2u-1rlNpnuCPGXKL7j94tRkXaY3aOVHOo7dmoEgLhEAhGY3-qihWDNQpEbAFlUz2i30az4afjo4a0CpWwObzWJWcmc571DDVC-f3H_XspcZY1L2lPTfuxUBFChlUEfw-lOb19QOQkmqD_MUQVSTod_yEw3aDsh3BaNMqXExJNvS83zygFmrv-1fMXL5lOezH5rH_z7qqWqonRbn_72-nwAxaz_r8KP9B_oV26lAs-QFDXL-9sMJGx6yYvOHx7NkW4obggMg0yHWigqZhFlW00
-->


An Extractor is composed of a trained predictor (black-box used as an oracle) and a set of discrete features
(some algorithms require a discrete dataset).
It provides two methods:
* extract: given a dataset it returns a theory;
* predict: return the predicted value applying the theory to the given data.

Currently, the supported extractors are:
* [CART](https://doi.org/10.1201/9781315139470),
straightforward extracts rules from both classification and regression decision trees;
* Classification:
  * [REAL](http://dx.doi.org/10.1016/B978-1-55860-335-6.50013-1) (Rule Extraction As Search),
  generates a rule for each sample in the dataset if the sample isn't covered yet.
  Before ending the extraction the rules set is optimized;
  * [Trepan](http://dx.doi.org/10.1016/B978-1-55860-335-6.50013-1),
  first it generates a decision tree using m-of-n expressions, than it extracts rule from it;
* Regression:
  * [ITER](http://dx.doi.org/10.1007/11823728_26),
  builds and iteratively expands hypercubes in the input space.
  Each cube holds the estimated value of the regression for the inputs that are inside the cube.
  Rules are generated from the cubes' dimensions;
  * [Gridex](http://dx.doi.org/10.1007/978-3-030-82017-6_2), coming soon.

## Users

### End users

PSyKE is a library that can be installed as python package by running:
```bash
pip install psyke
```

#### Requirements
* numpy 1.21.3+
* pandas 1.3.4+
* scikit-learn 1.0.1+
* 2ppy 0.3.2+
* skl2onnx 1.10.0+
* onnxruntime 1.9.0+
* parameterized 0.8.1+

Once installed one can create an extractor from a predictor
(e.g. Neural Networks, Support Vector Machines, K-Nearest Neighbor, Random Forests, etc.)
and from the dataset used to train the predictor.
**Note:** the predictor must have a method called `predict` to be properly used as oracle.

A brief example is presented in `demo.py` script.
Using sklearn iris dataset we train a K-Nearest Neighbor to predict the correct iris class.
Before training, we make the dataset discrete.
After that we create two different extractors: REAL and Trepan.
We output the extracted theory for both extractors.

REAL extracted rules:
```text
iris(PetalLength_0, PetalWidth_0, SepalLength_0, SepalWidth_0, setosa) :- '=<'(PetalWidth_0, 0.65).
iris(PetalLength_1, PetalWidth_1, SepalLength_1, SepalWidth_1, versicolor) :- ('>'(PetalLength_1, 4.87), '>'(SepalLength_1, 6.26)).
iris(PetalLength_2, PetalWidth_2, SepalLength_2, SepalWidth_2, versicolor) :- '>'(PetalWidth_2, 1.64).
iris(PetalLength_3, PetalWidth_3, SepalLength_3, SepalWidth_3, virginica) :- '=<'(SepalWidth_3, 2.87).
iris(PetalLength_4, PetalWidth_4, SepalLength_4, SepalWidth_4, virginica) :- in(SepalLength_4, [5.39, 6.26]).
iris(PetalLength_5, PetalWidth_5, SepalLength_5, SepalWidth_5, virginica) :- in(PetalWidth_5, [0.65, 1.64]).
```

Trepan extracted rules:
```text
iris(PetalLength_6, PetalWidth_6, SepalLength_6, SepalWidth_6, virginica) :- ('>'(PetalLength_6, 2.28), in(PetalLength_6, [2.28, 4.87])).
iris(PetalLength_7, PetalWidth_7, SepalLength_7, SepalWidth_7, versicolor) :- '>'(PetalLength_7, 2.28).
iris(PetalLength_8, PetalWidth_8, SepalLength_8, SepalWidth_8, setosa) :- true.
```


## Developers

Working with PSyKE codebase requires a number of tools to be installed:
* Python 3.9+
* JDK 11+ (please ensure the `JAVA_HOME` environment variable is properly configured)
* Git 2.20+

### Develop PSyKE with PyCharm

To participate in the development of PSyKE, we suggest the [PyCharm](https://www.jetbrains.com/pycharm/) IDE.

#### Importing the project

1. Clone this repository in a folder of your preference using `git_clone` appropriately
2. Open PyCharm
3. Select `Open`
4. Navigate your file system and find the folder where you cloned the repository
5. Click `Open`

### Developing the project

Contributions to this project are welcome. Just some rules:
* We use [git flow](https://github.com/nvie/gitflow), so if you write new features, please do so in a separate `feature/` branch
* We recommend forking the project, developing your stuff, then contributing back vie pull request
* Commit often
* Stay in sync with the `develop` branch (pull frequently if the build passes)
* Do not introduce low quality or untested code

#### Issue tracking
If you meet some problem in using or developing PSyKE, you are encouraged to signal it through the project
["Issues" section](https://github.com/psykei/psyke-python/issues) on GitHub.