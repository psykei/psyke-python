import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import svds
from scipy.special import betainc, gamma
import copy
import collections

from psyke import Clustering


class CLASSIX(Clustering):
    def __init__(self, minPts=0, radius=0.5,  group_merging_mode="distance", scale=1.5, reassign_outliers=True):     
        """
        Parameters:
            minPts : int, default=0
                Threshold used to find outlier clusters. 
                If reassign_outliers=True points of clusters that include less than minPts points are reallocated to thenearest cluster.

            radius : float, default=0.5 
                Maximum distance between the current starting point and datapoint to have the latter included in the starting point's group
            
            group_merging_mode : str, {'distance', 'density'}, default='distance'
                - 'density': merge two groups if the density of their intersection is greater than at least one of the two groups.               
                - 'distance': merge two groups if their starting points have a distance smaller than scale*radius.
             
            scale : float, default=1.5
                Parameter used for distance-based merging. The higher the parameter, the more groups are merged together.

            reassign_outliers : boolean, default=True
                If True, outliers are reassigned to the closest non-outlier cluster. If False they are marked with label -1
        """
       
        if group_merging_mode not in ("distance","density"):
            raise ValueError(f"Passed group_merging_mode='{group_merging_mode}' while only 'distance' and 'density' are allowed")
        self.minPts = minPts
        self.radius = radius
        self.group_merging_mode = group_merging_mode
        
        self.scale = scale 
        self.reassign_outliers = reassign_outliers

        self.points_group_labels = None
        self.starting_points_list = None
        self.cluster_labels = None
        self.merge_groups = None
        self.connected_pairs_edges = None
        self.reassigned_outlier_groups = None
    
    def fit(self, data):
        self.data = self.data_preparation(data)  #center and scale all data points
        self.points_group_labels, self.starting_points_list = self.aggregating() #aggregation labels (group labels)

        # Array of length=number of datapoints, all filled with True. It will be used to discard self.data points
        self.valid_data_index = np.full(self.data.shape[0], True) 
        self.cluster_labels = self.clustering(self.points_group_labels) 

        return self

    def data_preparation(self, data): 
        """Center and scale data so that 50% fall in the unitary ball, making mext=1
        
        Parameter:
            data (numpy.ndarray): datapoints on which the method has to be fit
        
        Returns:
            scaled_data (nmpy.ndarray): data centered and scaled
        """
        if not isinstance(data, np.ndarray):    #convert data to np.ndarray
            data = np.array(data)
            if len(data.shape) == 1:
                data = data.reshape(-1,1)
                
        data = data.astype('float64')

        self.mu = data.mean(axis=0)    #empirical mean value of each feature
        data = data - self.mu     #center all self.data points by taking off the empirical mean value of each feature
        distances_from_origin = norm(data, axis=1) # distance of each self.data point from 0
        self.median = np.median(distances_from_origin) # 50% of self.data points are within that radius
        if self.median == 0: # prevent zero-division
            self.median = 1
        scaled_data = data / self.median    # now 50% of self.data are in unit ball, so mext=1
        
        return scaled_data 

    def aggregating(self): 
        """Perform aggregation phase.

        Returns:         
            points_group_labels (numpy.ndarray) : 
                At index i group label of datapoint i after aggregation.
            
            starting_points_list (numpy.ndarray) : 
                Array of groups' starting points info.
                    at index [i,0] starting point index of group i
                    at index [i, 1] group i starting_point's alpha_score
                    at index [i, 2] number of group i elements
        """

        starting_points_list = list() # list that will contain groups' starting points
        
        if self.data.shape[1] > 1: # performing SVD to get alpha score of every datapoint
            U1, s1, _ = svds(self.data, k=1, return_singular_vectors=True)
            alpha_scores = U1[:,0]*s1[0]       #scores alpha-i

        else:   #if data has only one feature, svd is not needed
            alpha_scores = self.data[:,0]
        
        alpha_scores_ordered_indices = np.argsort(alpha_scores)     #ordered indeces of datapoints according to their alpha score

        current_label = 0   #starting labelling from 0. This label will increment moving to the next group
        points_group_labels = [-1] * self.data.shape[0]   # -1 means unassigned
        
        for i in range(self.data.shape[0]): #iterate over the whole dataset
            starting_point_index = alpha_scores_ordered_indices[i] # extracting starting point index
            if points_group_labels[starting_point_index] >= 0: # point is already assigned to a group
                continue        
            else:   #point is not assigned to a group, so it will be the starting point of a new one    
                current_group_center = self.data[starting_point_index,:]    #this it the first data point of the group. It will be used to compute the distance
                points_group_labels[starting_point_index] = current_label   #labelling the datapoint
                points_current_group_count = 1    #this new point determines a new group, with only this datapoint

            for j in alpha_scores_ordered_indices[i+1:]:     #scan all the remaining datapoints
                if points_group_labels[j] >= 0:  #point is already assigned
                    continue

                if (alpha_scores[j] - alpha_scores[starting_point_index] > self.radius):  #first far point, we have to exit the loop taking advantage of the ordered alpha scores
                    break       
       
                if norm(current_group_center - self.data[j,:], ord=2, axis=-1) <= self.radius:   # check if the point satisfies aggregation condition
                    points_group_labels[j] = current_label      #assign the label with the current group label
                    points_current_group_count += 1      #increase the number of points in the group

            starting_points_list.append([starting_point_index, alpha_scores[starting_point_index], points_current_group_count])  
            current_label += 1  # group assigned, moving to the next starting point

        return np.array(points_group_labels), np.array(starting_points_list)
    
    def merging(self):
        """Perform merging phase.
        
        Returns:

            labels_set : list
                Connected components of graph with groups as vertices.
            
            connected_pairs_store : list
                List for connected group labels.

        """

        class SET:
            """Disjoint-set data structure."""
            def __init__(self, data):
                self.data = data
                self.parent = self
                       
        def findParent(s):
            """Recursive procedure to find parent of a node and return it."""
            if (s.data != s.parent.data) :
                s.parent = findParent((s.parent))
            return s.parent

        def mergeRoots(s1, s2):
            """Merge the roots of two node. Returns nothing."""
            parent_of_s1 = findParent(s1)
            parent_of_s2 = findParent(s2)

            if (parent_of_s1.data != parent_of_s2.data) :
                findParent(s1).parent = parent_of_s2

        def checkOverlap(starting_point_1, starting_point_2, radius, scale = 1):
            """Check if two groups overlap, return True or False accordingly"""
            return norm(starting_point_1 - starting_point_2, ord=2, axis=-1) <= 2 * scale * radius

        def computeIntersectionDensity(starting_point_1, starting_point_2, radius, samples_number):
            """Calculate the density of intersection"""
            in_volume = computeIntersectionVolume(starting_point_1, starting_point_2, radius)
            return samples_number / in_volume

        def computeIntersectionVolume(starting_point_1, starting_point_2, radius):
            """Returns the volume of the intersection of two n-dimensional spheres with the specified radius."""

            dimension = starting_point_1.shape[0]
            center_distance = norm(starting_point_1 - starting_point_2, ord=2, axis=-1) # the distance between the two groups

            if center_distance > 2*radius:
                return 0
            c = center_distance / 2
            return np.pi**(dimension/2)/gamma(dimension/2 + 1)*(radius**dimension)*betainc((dimension + 1)/2, 1/2, 1 - c**2/radius**2)

        connected_pairs = [SET(i) for i in range(self.starting_points_list.shape[0])]    #creating a SET for every starting point
        connected_pairs_store = []  #list where each element [i,j] is an arch from i to j in the connected componenets merging
        if self.group_merging_mode == "density":
            volume = np.pi**(self.data.shape[1]/2) * self.radius**self.data.shape[1] / gamma(self.data.shape[1]/2+1)
        else:
            volume = None
            
        for i in range(self.starting_points_list.shape[0]):      #loop through starting points    
            first_starting_point =  self.data[int(self.starting_points_list[i, 0])] # extract the datapoint corresponding to the index of the starting point
            successive_starting_points = self.data[self.starting_points_list[i+1:, 0].astype(int)] # get the rest of starting points from position i+1
            
            #mask with starting points indeces. Initialized to the range of all the successive starting points indeces
            close_starting_points = np.arange(i+1, self.starting_points_list.shape[0], dtype=int)     
            current_alpha_scores_ordered = self.starting_points_list[i:, 1]
            
            if self.group_merging_mode == "density":                    # calculate the density
                index_overlap = norm(successive_starting_points - first_starting_point, ord=2, axis=-1) <= 2*self.radius # 2*distance_scale*radius
                close_starting_points = close_starting_points[index_overlap]

                if not np.any(index_overlap):
                    continue

                points_in_first_ball = norm(self.data-first_starting_point, ord=2, axis=-1) <= self.radius   # compute array of True or False to get the points in the Ball centered in first_starting_point
                density_first_ball = np.count_nonzero(points_in_first_ball) / volume    #compute density of the first ball

                for j in close_starting_points.astype(int):
                    next_starting_point = self.data[int(self.starting_points_list[j, 0])] 

                    points_in_next_ball = norm(self.data-next_starting_point, ord=2, axis=-1) <= self.radius
                    density_next_ball = np.count_nonzero(points_in_next_ball) / volume  #compute density of the second ball

                    if checkOverlap(first_starting_point, next_starting_point, radius=self.radius): #check if the two balls overlap
                        points_in_intersection_count = np.count_nonzero(points_in_first_ball & points_in_next_ball)    #find points that are in the intersection of balls
                        density_intersection = computeIntersectionDensity(first_starting_point, next_starting_point, radius=self.radius, samples_number=points_in_intersection_count)
                        if density_intersection >= density_first_ball or density_intersection >= density_next_ball: 
                            mergeRoots(connected_pairs[i], connected_pairs[j])
                            connected_pairs_store.append([i, j])
            else: # group_merging_mode="distance": 

                index_overlap=np.full(successive_starting_points.shape[0], False, dtype=bool)   # Initialize an array of all false 
                for k in range(successive_starting_points.shape[0]):
                    candidate = successive_starting_points[k]
                    
                    if (current_alpha_scores_ordered[k+1] - current_alpha_scores_ordered[0] > self.scale*self.radius):
                        break
                        
                    if norm(candidate - first_starting_point, ord=2, axis=-1) <= self.scale*self.radius:
                        index_overlap[k] = True
                            
                if not np.any(index_overlap): # if no overlapping occurs
                    continue 

                close_starting_points = close_starting_points[index_overlap]    #all starting points from index i+1 that have a mask value True
                      
                for j in close_starting_points:
                    # two groups merge when their starting points distance is smaller than scale*radius
                    mergeRoots(connected_pairs[i], connected_pairs[j])   # merge roots of starting points
                    connected_pairs_store.append([i, j])    # append indeces of merged componenents
            
        labels = [findParent(i).data for i in connected_pairs]
        labels_set = list()
        for i in np.unique(labels):
            labels_set.append(np.where(labels == i)[0].tolist())    
        return labels_set, connected_pairs_store  

    def outlier_filter(self, old_cluster_count, min_samples_rate=0.1): 
        """Filter outliers in terms of min_samples"""

        if self.minPts == None:
            min_samples = min_samples_rate*sum(old_cluster_count.values())
        else:
            min_samples= self.minPts
        return [i[0] for i in old_cluster_count.items() if i[1] < min_samples]

    def reassign_labels(self, labels, old_cluster_count):
        sorted_dict = sorted(old_cluster_count.items(), key=lambda x: x[1], reverse=True)

        clabels = copy.deepcopy(labels)
        for i in range(len(sorted_dict)):
            clabels[labels == sorted_dict[i][0]]  = i
        return clabels

    def clustering(self, points_group_labels):
        labels = copy.deepcopy(points_group_labels) #deep copy of aggregation labels (group labels)
        self.merge_groups, self.connected_pairs_edges = self.merging()
        max_label = max(labels) + 1 # starting label for clusters

        # fully merging griups to get clusters 
        for sublabels in self.merge_groups: 
            for j in sublabels:
                labels[labels == j] = max_label
            max_label = max_label + 1
        
        # extracting the clusters with very rare number of objects as potential "noises"
        old_cluster_count = collections.Counter(labels)
        
        if self.minPts >= 1:
            potential_noise_labels = self.outlier_filter(old_cluster_count=old_cluster_count,
                                                         min_samples_rate=self.minPts) 
        
            if len(potential_noise_labels) > 0:
                for i in np.unique(potential_noise_labels):
                    labels[labels == i] = max_label     #mark datapoints with max_label to set them as outliers

                self.valid_data_index = labels != max_label     # mark outliers as not valid datapoints in the index
                valid_group_labels = points_group_labels[self.valid_data_index]     #extracting non outlier starting points

                self.label_change = dict(zip(valid_group_labels, labels[self.valid_data_index])) # store the mapping between valid group labels and cluster labels
                
                unique_outliers_group_labels = np.unique(points_group_labels[~self.valid_data_index]) # get unique labels of group marked as outlier
                unique_valid_group_labels = np.unique(valid_group_labels)   # get unique labels of non outliers groups
                valid_starting_points_list = self.starting_points_list[unique_valid_group_labels]   # list of starting points of non outlier groups

                if self.reassign_outliers and len(self.data[valid_starting_points_list[:, 0].astype(int)])!=0:  # reassign outliers to the closest groups
                    for outlier_label in unique_outliers_group_labels:
                        closest_group = np.argmin(norm(self.data[valid_starting_points_list[:, 0].astype(int)] - self.data[int(self.starting_points_list[outlier_label, 0])], axis=1, ord=2))
                        labels[points_group_labels == outlier_label] = self.label_change[unique_valid_group_labels[closest_group]]

                else:    # mark outliers with label -1
                    labels[np.isin(points_group_labels, unique_outliers_group_labels)] = -1 
        
        # reassign labels so that they start from 0 (and eventually there is -1 for outliers)
        labels = self.reassign_labels(labels=labels,
                                     old_cluster_count=old_cluster_count) 
        self.label_change = dict(zip(points_group_labels, labels)) # mapping between group labels and cluster labels
            
        return labels 
    
    def predict(self, data_to_predict):
        """Predict the cluster to which every datapoint belongs.
        Parameter:
            data_to_predict : numpy.ndarray
                The ndarray-like input of shape (n_samples,)
        Returns:
            labels : numpy.ndarray
                The predicted clustering labels.
        """

        labels = list()
        
        data_to_predict = (data_to_predict - self.mu) / self.median
        indices = self.starting_points_list[:,0].astype(int)
        for i in range(len(data_to_predict)):
            splabel = np.argmin(norm(self.data[indices] - data_to_predict[i], axis=1, ord=2))
            labels.append(self.label_change[splabel])
  
        return labels
    
    def explain(self, dataIndex = None, generalExplanation = True, printStartingPoints = True):
        """Explain the clustering. 
        Parameter:
            dataIndex : int (default None)
                Index of the datapoint for which an explaination is desired.
                If None, no specific explanation is given.
            generalExplanation : bool (Deafault True)
                If True, a general explanation of the clustering process is printed.
            printStartingPoints : bool (Default True)
                If True, the list of all the starting points along with their alpha scores is printed.
        """

        if not (dataIndex == None or isinstance(dataIndex, int)):
            raise ValueError("dataIndex can only be an integer or None")
        
        data_size, feat_dim = self.data.shape

        if generalExplanation:
            print(f"\tGeneralExplanation")
            print("""A clustering of {length:.0f} data points with {dim:.0f} features has been performed. """.format(length=data_size, dim=feat_dim))
            print("""The radius parameter was set to {r:.2f} and MinPts was set to {minPts:.0f}. """.format(r=self.radius, minPts=self.minPts))
            print("""As the provided data has been scaled by a factor of 1/{scl:.2f}, data points within a radius of R={tol:.2f}*{scl:.2f}={tolscl:.2f} were aggregated into groups. """.format(
                scl=self.median, tol=self.radius, tolscl=self.median*self.radius
            ))
            print("""This resulted in {groups:.0f} groups, each uniquely associated with a starting point. """.format(groups=self.starting_points_list.shape[0]))
            print("""These {groups:.0f} groups were subsequently merged into {num_clusters:.0f} clusters resulting in the following mapping groups --> cluster:""".format(groups=self.starting_points_list.shape[0], num_clusters=len(np.unique(self.cluster_labels))))
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
            object1 = self.data[dataIndex] # self.data has been normalized
            object1_unscaled = (object1 * self.median) + self.mu #rescaling up the object
            agg_label1 = self.points_group_labels[dataIndex] # get the group index for object1

            
            
            cluster_label1 = self.label_change[agg_label1]
            group_s_p_data_index = self.starting_points_list[agg_label1][0].astype(int)
            group_s_p = self.data[group_s_p_data_index]
            group_s_p_unscaled = (group_s_p * self.median) + self.mu

            print("""
    Specific data point explanation            
The data point of index %(index1)s is in group %(agg_id)i, which has been merged into cluster #%(m_c)i.
Group %(agg_id)i is represented by starting point of index %(agg_id)i, which is at index %(indexsp)s in the dataset."""% {
                        "index1":dataIndex, "agg_id":agg_label1, "m_c":cluster_label1, "indexsp": group_s_p_data_index, 
                    }
                )

            if printStartingPoints:
                print(f"Starting point {agg_label1} scaled coordinates: {group_s_p}")
                print(f"Starting point {agg_label1} unscaled coordinates: {group_s_p_unscaled}")

        if printStartingPoints:
            print("\nBelow the list of all the groups starting points (unscaled):\n")
            for index, sp in enumerate(starting_points_unscaled):
                print(f"Starting point {index} has alpha score = {starting_points_alpha_scores[index]:.3f} and coordinates {sp}")


        
                