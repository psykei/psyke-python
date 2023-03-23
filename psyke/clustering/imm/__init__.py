from typing import Iterable

from scipy.optimize import minimize
from sklearn.cluster import KMeans
import numpy as np

from psyke import Clustering


class Node:

    def __init__(self):
        self.reference = None
        self.left = None
        self.right = None
        self.feature = None
        self.value = None
        self.cluster = None

    def isLeaf(self):
        return self.cluster is not None


class IMM(Clustering):

    def __init__(self, k):
        super().__init__()
        self.k = k  # number of clusters
        self.tree = None  # will be the created tree
        self.nodeCount = 0

    def fit(self, x):

        # perform kmeans clustering on dataset
        clt = KMeans(n_clusters=self.k, random_state=0, n_init=10)
        clusterK = clt.fit(x)

        u = np.array(clusterK.cluster_centers_)  # clusters' centers
        y = np.array(clusterK.labels_)  # clusters' labels

        self.tree = self.build_tree(x, y, u)

    def build_tree(self, x, y, u):

        # check if array is homogenous (i.e. all the elements are equal)
        if np.all(y == y[0]):
            leaf = Node()
            leaf.reference = self.nodeCount
            self.nodeCount += 1
            leaf.cluster = y[0]
            return leaf

        else:
            dataDim = len(x)
            n_features = len(x[0])

            # populate arrays of l and r
            l = np.zeros(n_features)
            r = np.zeros(n_features)

            for i in range(n_features):
                arr = np.array([u[y[j]][i] for j in range(dataDim)])  # getting the i-th feature from the y[j]-th center
                l[i] = np.amin(arr)
                r[i] = np.amax(arr)

            optimizedMistakes = []  # at index i will it contain minimalOfMistakes on feature i
            # at index i will contain the value of theta that ensures minimalOfMistakes[i] mistakes for feature i
            optimalTheta = []
            initialGuesses = np.vstack([l[:], r[:]]).mean(axis=0)  # initial guesses

            for i in range(n_features):
                optimizedResult = minimize(fun=self.sum_mistakes, x0=(initialGuesses[i]),
                                           args=(x, u, y, i), bounds=[(l[i], r[i])]
                                           )

                optimizedMistakes.append(optimizedResult.fun)  # value of the objective function
                optimalTheta.append(optimizedResult.x[0])  # best theta

            i = np.argmin(optimizedMistakes)
            theta = optimalTheta[i]

            M = []
            L = []
            R = []

            for j in range(dataDim):
                if self.mistake(x[j], u[y[j]], i, theta) == 1:
                    M.append(j)
                elif x[j][i] <= theta:
                    L.append(j)
                elif x[j][i] > theta:
                    R.append(j)

            leftx = []
            lefty = []

            for e in range(dataDim):
                if e in L:
                    leftx.append(x[e])
                    lefty.append(y[e])

            rightx = []
            righty = []

            for e in range(dataDim):
                if e in R:
                    rightx.append(x[e])
                    righty.append(y[e])

            node = Node()
            node.reference = self.nodeCount
            self.nodeCount += 1
            node.feature = i
            node.value = theta
            node.left = self.build_tree(leftx, lefty, u)
            node.right = self.build_tree(rightx, righty, u)

            return node

    def mistake(self, x, u, i, theta):
        return 1 if (x[i] <= theta) != (u[i] <= theta) else 0

    def sum_mistakes(self, theta, x, u, y, i):
        return sum([self.mistake(x[j], u[y[j]], i, theta) for j in range(len(x))])

    def _predict(self, x):
        if self.tree is None:
            raise TypeError('Model is untrained.')
        elif isinstance(x, Iterable):
            return [self.find_cluster(self.tree, xx) for xx in x]
        else:
            return self.find_cluster(self.tree, x)

    def predict_on_more_data(self, x):
        if self.tree is None:
            raise TypeError('Model is untrained.')
        return [self.find_cluster(self.tree, x[i]) for i in range(len(x))]

    def find_cluster(self, tree, x):
        return tree.cluster if tree.cluster is not None else \
            self.find_cluster(tree.left if x[tree.feature] <= tree.value else tree.right, x)

    def explore(self, node):
        nodes = [node]
        edges = []

        if node.cluster is None:
            left_subtree, right_subtree = self.explore(node.left), self.explore(node.right)
            nodes += left_subtree[0] + right_subtree[0]
            edges = [(node, node.left), (node, node.right)] + left_subtree[1] + right_subtree[1]

        return nodes, edges
