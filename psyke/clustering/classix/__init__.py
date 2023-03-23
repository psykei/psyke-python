import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import svds
from scipy.special import betainc, gamma
import copy
import collections

from psyke import Clustering


class CLASSIX(Clustering):
    def __init__(self, minPts: int = 0, radius: float = 0.5, group_merging_mode: str = "distance",
                 scale: float = 1.5, reassign_outliers: bool = True):
        """
        :param minPts: Threshold used to find outlier clusters. If reassign_outliers=True points of clusters that
                       include less than minPts points are reallocated to thenearest cluster.

        :param radius: Maximum distance between the current starting point and datapoint to have the latter included in
                       the starting point's group
            
        :param group_merging_mode: Strategy to merge two groups.
                        If 'density': merge if density of their intersection is greater than at least one of them.
                        If 'distance': merge if their starting points have a distance smaller than scale*radius.
             
        :param scale: Used for distance-based merging. The higher the parameter, the more groups are merged together.

        :param reassign_outliers: If True, outliers are reassigned to the closest non-outlier cluster.
                                  If False they are marked with label -1
        """

        super().__init__()
        if group_merging_mode not in ("distance", "density"):
            raise ValueError("Only 'distance' and 'density' group_merging_mode are allowed")
        self.minPts = minPts
        self.radius = radius
        self.group_merging_mode = group_merging_mode
        self.scale = scale
        self.reassign_outliers = reassign_outliers

        self.data = None
        self.mu = 0.0
        self.median = 1.0

        self.points_group_labels = None
        self.starting_points_list = None
        self.cluster_labels = None
        self.label_change = None

    def fit(self, data):
        self.data = self.data_preparation(np.array(data).astype('float64'))
        self.aggregate()
        self.clustering()
        return self

    def data_preparation(self, data):
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        self.mu = data.mean(axis=0)
        data -= self.mu
        median = np.median(norm(data, axis=1))
        self.median = median if median != 0.0 else 1.0
        return data / self.median

    def aggregate(self):
        starting_points_list = list()

        if self.data.shape[1] > 1:
            u1, s1, _ = svds(self.data, k=1, return_singular_vectors=True)
            alpha_scores = u1[:, 0] * s1[0]
        else:
            alpha_scores = self.data[:, 0]

        alpha_scores_ordered_indices = np.argsort(alpha_scores)
        current_label = 0
        points_group_labels = [-1] * self.data.shape[0]

        for i in range(self.data.shape[0]):
            starting_point_index = alpha_scores_ordered_indices[i]
            if points_group_labels[starting_point_index] >= 0:
                continue

            current_group_center = self.data[starting_point_index, :]
            points_group_labels[starting_point_index] = current_label
            points_current_group_count = 1

            for j in alpha_scores_ordered_indices[i + 1:]:
                if points_group_labels[j] >= 0:
                    continue
                if alpha_scores[j] - alpha_scores[starting_point_index] > self.radius:
                    break
                if norm(current_group_center - self.data[j, :], ord=2, axis=-1) <= self.radius:
                    points_group_labels[j] = current_label
                    points_current_group_count += 1

            starting_points_list.append([starting_point_index, alpha_scores[starting_point_index],
                                         points_current_group_count])
            current_label += 1

        self.points_group_labels = np.array(points_group_labels)
        self.starting_points_list = np.array(starting_points_list)

    def merging(self):
        class SET:
            def __init__(self, data):
                self.data = data
                self.parent = self

        def findParent(s):
            return s.parent if s.data == s.parent.data else findParent(s.parent)

        def mergeRoots(s1, s2):
            parent_of_s2 = findParent(s2)
            if findParent(s1).data != parent_of_s2.data:
                findParent(s1).parent = parent_of_s2

        def checkOverlap(starting_point_1, starting_point_2, radius, scale=1):
            return norm(starting_point_1 - starting_point_2, ord=2, axis=-1) <= 2 * scale * radius

        def computeIntersectionDensity(starting_point_1, starting_point_2, radius, samples_number):
            return samples_number / computeIntersectionVolume(starting_point_1, starting_point_2, radius)

        def computeIntersectionVolume(starting_point_1, starting_point_2, radius):
            dimension = starting_point_1.shape[0]
            center_distance = norm(starting_point_1 - starting_point_2, ord=2, axis=-1)

            return 0 if center_distance > 2 * radius else np.pi ** (dimension / 2) / gamma(dimension / 2 + 1) * (
                    radius ** dimension) * betainc((dimension + 1) / 2, 0.5, 1 - (center_distance / 2 / radius) ** 2)

        connected_pairs = [SET(i) for i in range(self.starting_points_list.shape[0])]

        volume = np.pi ** (self.data.shape[1] / 2) * self.radius ** self.data.shape[1] / gamma(
            self.data.shape[1] / 2 + 1) if self.group_merging_mode == "density" else None

        for i in range(self.starting_points_list.shape[0]):
            first_starting_point = self.data[int(self.starting_points_list[i, 0])]
            succ_starting_points = self.data[self.starting_points_list[i + 1:, 0].astype(int)]

            close_starting_points = np.arange(i + 1, self.starting_points_list.shape[0], dtype=int)
            current_alpha_scores_ordered = self.starting_points_list[i:, 1]

            if self.group_merging_mode == "density":
                index_overlap = norm(succ_starting_points - first_starting_point, ord=2, axis=-1) <= 2 * self.radius
                close_starting_points = close_starting_points[index_overlap]

                if not np.any(index_overlap):
                    continue

                points_in_first_ball = norm(self.data - first_starting_point, ord=2, axis=-1) <= self.radius
                density_first_ball = np.count_nonzero(points_in_first_ball) / volume

                for j in close_starting_points.astype(int):
                    next_starting_point = self.data[int(self.starting_points_list[j, 0])]

                    points_in_next_ball = norm(self.data - next_starting_point, ord=2, axis=-1) <= self.radius
                    density_next_ball = np.count_nonzero(points_in_next_ball) / volume

                    if checkOverlap(first_starting_point, next_starting_point, radius=self.radius):
                        points_in_intersection_count = np.count_nonzero(points_in_first_ball & points_in_next_ball)
                        density_intersection = computeIntersectionDensity(first_starting_point, next_starting_point,
                                                                          radius=self.radius,
                                                                          samples_number=points_in_intersection_count)
                        if density_intersection >= density_first_ball or density_intersection >= density_next_ball:
                            mergeRoots(connected_pairs[i], connected_pairs[j])
            else:
                index_overlap = np.full(succ_starting_points.shape[0], False, dtype=bool)
                for k in range(succ_starting_points.shape[0]):
                    candidate = succ_starting_points[k]

                    if current_alpha_scores_ordered[k + 1] - current_alpha_scores_ordered[0] > self.scale * self.radius:
                        break

                    if norm(candidate - first_starting_point, ord=2, axis=-1) <= self.scale * self.radius:
                        index_overlap[k] = True

                if not np.any(index_overlap):
                    continue

                close_starting_points = close_starting_points[index_overlap]

                for j in close_starting_points:
                    mergeRoots(connected_pairs[i], connected_pairs[j])

        labels = [findParent(i).data for i in connected_pairs]
        labels_set = list()
        for i in np.unique(labels):
            labels_set.append(np.where(labels == i)[0].tolist())
        return labels_set

    def outlier_filter(self, old_cluster_count, min_samples_rate=0.1):
        min_samples = min_samples_rate * sum(old_cluster_count.values()) if self.minPts is None else self.minPts
        return [i[0] for i in old_cluster_count.items() if i[1] < min_samples]

    def reassign_labels(self, labels, old_cluster_count):
        sorted_dict = sorted(old_cluster_count.items(), key=lambda x: x[1], reverse=True)
        clabels = copy.deepcopy(labels)
        for i in range(len(sorted_dict)):
            clabels[labels == sorted_dict[i][0]] = i
        self.cluster_labels = clabels

    def clustering(self):
        labels = copy.deepcopy(self.points_group_labels)
        merge_groups = self.merging()
        max_label = max(labels) + 1

        for sublabels in merge_groups:
            for j in sublabels:
                labels[labels == j] = max_label
            max_label = max_label + 1

        old_cluster_count = collections.Counter(labels)

        if self.minPts >= 1:
            potential_noise_labels = self.outlier_filter(old_cluster_count=old_cluster_count,
                                                         min_samples_rate=self.minPts)

            if len(potential_noise_labels) > 0:
                for i in np.unique(potential_noise_labels):
                    labels[labels == i] = max_label

                valid_data_index = labels != max_label
                valid_group_labels = self.points_group_labels[valid_data_index]

                self.label_change = dict(zip(valid_group_labels, labels[valid_data_index]))

                unique_outliers_group_labels = np.unique(self.points_group_labels[~valid_data_index])
                unique_valid_group_labels = np.unique(valid_group_labels)
                valid_starting_points_list = self.starting_points_list[unique_valid_group_labels]

                if self.reassign_outliers and len(self.data[valid_starting_points_list[:, 0].astype(int)]) != 0:
                    for outlier_label in unique_outliers_group_labels:
                        closest_group = np.argmin(norm(self.data[valid_starting_points_list[:, 0].astype(int)] -
                                                       self.data[int(self.starting_points_list[outlier_label, 0])],
                                                       axis=1, ord=2))
                        labels[self.points_group_labels == outlier_label] = \
                            self.label_change[unique_valid_group_labels[closest_group]]

                else:
                    labels[np.isin(self.points_group_labels, unique_outliers_group_labels)] = -1

        self.reassign_labels(labels=labels, old_cluster_count=old_cluster_count)
        self.label_change = dict(zip(self.points_group_labels, self.cluster_labels))

    def _predict(self, data_to_predict, mapping: dict[str: int] = None):
        data_to_predict = (data_to_predict - self.mu) / self.median
        indices = self.starting_points_list[:, 0].astype(int)
        return [self.label_change[np.argmin(norm(self.data[indices] - d, axis=1, ord=2))] for d in data_to_predict]

    def explain(self, dataIndex: int = None, generalExplanation: bool = True, printStartingPoints: bool = True):
        """Explain the clustering. 
        :param dataIndex: Index of the datapoint for which an explaination is desired.
                          If None, no specific explanation is given.
        :param generalExplanation: If True, a general explanation of the clustering process is printed.
        :param printStartingPoints: If True, the list of all starting points along with their alpha scores is printed.
        """

        if not (dataIndex is None or isinstance(dataIndex, int)):
            raise ValueError("dataIndex can only be an integer or None")

        data_size, feat_dim = self.data.shape

        if generalExplanation:
            print(f"\tGeneralExplanation\n"
                  f"A clustering of {data_size:.0f} data points with {feat_dim:.0f} features has been performed.\n"
                  f"The radius parameter was set to {self.radius:.2f} and MinPts was set to {self.minPts:.0f}.\n"
                  f"As the provided data has been scaled by a factor of 1/{self.median:.2f}, data points within a "
                  f"radius of R={self.radius:.2f}*{self.median:.2f}={self.median * self.radius:.2f} were aggregated "
                  f"into groups.\nThis resulted in {self.starting_points_list.shape[0]:.0f} groups, each uniquely "
                  f"associated with a starting point.\nThese {self.starting_points_list.shape[0]:.0f} groups were "
                  f"subsequently merged into {len(np.unique(self.cluster_labels)):.0f} clusters resulting in the "
                  f"following mapping groups --> cluster:")
            merging = {cluster: [] for cluster in sorted(set(self.label_change.values()))}
            for group, cluster in self.label_change.items():
                merging[cluster].append(group)
            merging = {cluster: sorted(groups) for cluster, groups in merging.items()}

            for cluster, groups in merging.items():
                print(f"Groups {groups} --> Cluster {cluster}")

        starting_points = self.data[self.starting_points_list[:, 0].astype(int)]
        starting_points_unscaled = (starting_points * self.median) + self.mu
        starting_points_alpha_scores = self.starting_points_list[:, 1]

        if isinstance(dataIndex, int):
            agg_label1 = self.points_group_labels[dataIndex]

            cluster_label1 = self.label_change[agg_label1]
            group_s_p_data_index = self.starting_points_list[agg_label1][0].astype(int)
            group_s_p = self.data[group_s_p_data_index]
            group_s_p_unscaled = (group_s_p * self.median) + self.mu

            print(f"\tSpecific data point explanation\nThe data point of index {dataIndex} is in group {agg_label1}, "
                  f"which has been merged into cluster #{cluster_label1}.\nGroup {agg_label1} is represented by "
                  f"starting point of index {agg_label1}, which is at index {group_s_p_data_index} in the dataset.")

            if printStartingPoints:
                print(f"Starting point {agg_label1} scaled coordinates: {group_s_p}")
                print(f"Starting point {agg_label1} unscaled coordinates: {group_s_p_unscaled}")

        if printStartingPoints:
            print("\nBelow the list of all the groups starting points (unscaled):\n")
            for index, sp in enumerate(starting_points_unscaled):
                print(f"Starting point {index} has alpha score = {starting_points_alpha_scores[index]:.3f} and "
                      f"coordinates {sp}")
