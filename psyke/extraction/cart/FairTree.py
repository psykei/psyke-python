import numpy as np
from collections import Counter

from sklearn.metrics import accuracy_score, r2_score


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class FairTree:
    def __init__(self, max_depth=3, max_leaves=None, criterion=None, min_samples_split=2, lambda_penalty=0.0,
                 protected_attr=None):
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.min_samples_split = min_samples_split
        self.lambda_penalty = lambda_penalty
        self.protected_attr = protected_attr
        self.criterion = criterion
        self.root = None
        self.n_leaves = 0
        self.quality_function = None

    def fit(self, X, y):
        self.n_leaves = 0
        self.root = self._grow_tree(X, y, depth=0)
        while self.n_leaves > self.max_leaves:
            self.prune_least_important_leaf(X, y)
            self.n_leaves -= 1
        return self

    @staticmethod
    def _estimate_output(y):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for _, x in X.iterrows()])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _grow_tree(self, X, y, depth):
        if depth >= self.max_depth or X.shape[0] < self.min_samples_split or len(set(y.values.flatten())) == 1 or \
                (self.max_leaves is not None and self.n_leaves >= self.max_leaves):
            self.n_leaves += 1
            return Node(value=self._estimate_output(y))

        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            self.n_leaves += 1
            return Node(value=self._estimate_output(y))

        left_idxs = X[best_feature] <= best_threshold
        right_idxs = X[best_feature] > best_threshold

        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        return Node(best_feature, best_threshold, left, right)

    @staticmethod
    def generate_thresholds(X, y):
        sorted_indices = np.argsort(X)
        X = np.array(X)[sorted_indices]
        y = np.array(y)[sorted_indices]
        # X = np.array(np.unique(np.unique(list(zip(X, y)), axis=0)[:, 0]), dtype=float)
        return np.array([(X[:-1][i] + X[1:][i]) / 2.0 for i in range(len(X) - 1) if y[i] != y[i + 1]])

    def _best_split(self, X, y):
        best_gain = -float('inf')
        split_idx, split_threshold = None, None

        for feature in [feature for feature in X.columns if feature not in self.protected_attr]:
            # for threshold in self.generate_thresholds(X[feature], y):
            for threshold in np.unique(np.quantile(X[feature], np.linspace(0, 1, num=25))):
                left_idxs = X[feature] <= threshold
                right_idxs = X[feature] > threshold

                if left_idxs.sum() == 0 or right_idxs.sum() == 0:
                    continue

                gain = self._fair_gain(y, left_idxs, right_idxs, X[self.protected_attr])

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature
                    split_threshold = threshold
        return split_idx, split_threshold

    @staticmethod
    def _disparity(group):
        counts = Counter(group)
        if len(counts) <= 1:
            return 0.0
        values = np.array(list(counts.values())) / len(group)
        return np.abs(values[0] - values[1])

    def _fair_gain(self, y, left_idx, right_idx, protected):
        child = len(y[left_idx]) / len(y) * self.quality_function(y[left_idx]) + \
                len(y[right_idx]) / len(y) * self.quality_function(y[right_idx])
        info_gain = self.quality_function(y) - child
        penalty = self._disparity(protected[left_idx]) + self._disparity(protected[right_idx])
        return info_gain - self.lambda_penalty * penalty

    @staticmethod
    def _match_path(x, path):
        for node, left in path:
            if left and x[node.feature] > node.threshold:
                return False
            if not left and x[node.feature] <= node.threshold:
                return False
        return True

    @staticmethod
    def candidates(node, parent=None, is_left=None, path=[]):
        if node is None or node.is_leaf_node():
            return []
        leaves = []
        if node.left.is_leaf_node() and node.right.is_leaf_node():
            leaves.append((node, parent, is_left, path))
        leaves += FairTreeClassifier.candidates(node.left, node, True, path + [(node, True)])
        leaves += FairTreeClassifier.candidates(node.right, node, False, path + [(node, False)])
        return leaves

    def prune_least_important_leaf(self, X, y):
        best_score = -np.inf
        best_prune = None

        for node, parent, is_left, path in self.candidates(self.root):
            original_left = node.left
            original_right = node.right

            merged_y = y[(X.apply(lambda x: self._match_path(x, path), axis=1))]
            if len(merged_y) == 0:
                continue
            new_value = self._estimate_output(merged_y)
            node.left = node.right = None
            node.value = new_value

            score = self.score(X, y)
            if score >= best_score:
                best_score = score
                best_prune = (node, new_value)

            node.left, node.right, node.value = original_left, original_right, None

        if best_prune:
            best_prune[0].left = best_prune[0].right = None
            best_prune[0].value = best_prune[1]


class FairTreeClassifier(FairTree):
    def __init__(self, max_depth=3, max_leaves=None, criterion='entropy', min_samples_split=2, lambda_penalty=0.0,
                 protected_attr=None):
        super().__init__(max_depth, max_leaves, criterion, min_samples_split, lambda_penalty, protected_attr)
        self.quality_function = self._gini if self.criterion == 'gini' else self._entropy

    @staticmethod
    def _estimate_output(y):
        return Counter(y.values.flatten()).most_common(1)[0][0]

    def score(self, X, y):
        return accuracy_score(y.values.flatten(), self.predict(X))

    @staticmethod
    def _entropy(y):
        ps = np.unique(y, return_counts=True)[1] / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    @staticmethod
    def _gini(y):
        return 1.0 - np.sum(np.unique(y, return_counts=True)[1] / len(y)**2)


class FairTreeRegressor(FairTree):
    def __init__(self, max_depth=3, max_leaves=None, criterion='mse', min_samples_split=2, lambda_penalty=0.0,
                 protected_attr=None):
        super().__init__(max_depth, max_leaves, criterion, min_samples_split, lambda_penalty, protected_attr)
        self.quality_function = self._mse

    @staticmethod
    def _estimate_output(y):
        return np.mean(y.values.flatten())

    def score(self, X, y):
        return r2_score(y.values.flatten(), self.predict(X))

    @staticmethod
    def _mse(y):
        y = y.values.flatten().astype(float)
        return np.mean((y - np.mean(y))**2)
