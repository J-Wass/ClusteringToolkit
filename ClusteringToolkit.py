import random
import numpy as np
import math

class KMeans:
    def __init__(self, k_clusters):
        if not (type(k_clusters) is int):
            raise TypeError('Expected k_clusters to be a valid integer')
            return
        if not k_clusters > 1:
            raise ValueError('k_clusters must be greater than 1')
            return
        self.k_clusters = k_clusters
        self.centroids = {}
        self.data = []
        self.clustered_data = []

    # takes in an array of feature arrays, builds clusters after validating data
    def fit(self, data):
        if not (type(data) is list):
            raise TypeError('Expected data to be a valid multidimensional array')
            return
        if len(data) <= self.k_clusters:
            raise ValueError('Size of data must be bigger than number of clusters!')
            return
        self.data = data
        self.__build_clusters()
        return list(map(lambda x: x[1],self.clustered_data)) #second index in clustered_data tuple is the class

    # creates k clusters using several helper methods
    def __build_clusters(self):
        converged = False
        self.__initialize_centroids()
        while not converged:
            # assign each data point to some cluster
            for data_point in self.data:
                closest_centroid = self.__get_closest_centroid(data_point)
                self.clustered_data.append((data_point, closest_centroid))
            converged = self.__rewrite_centroids()
            if not converged:
                self.clustered_data = []

    # k-means++ algorithm for optimal initial centroids
    def __initialize_centroids(self):
        initial_centroid = random.sample(self.data, 1)
        cluster = 0
        self.centroids[cluster] = initial_centroid
        cluster += 1
        while len(self.centroids) < self.k_clusters:
            # minimum distance vector of each centroid to each point
            D2 = np.array([min([np.linalg.norm(np.asarray(x)-np.asarray(self.centroids.get(c)))**2 for c in self.centroids]) for x in self.data])
            probability_distribution = D2/D2.sum()
            cumulative_prob_disto = probability_distribution.cumsum()
            r = random.random()
            index = np.where(cumulative_prob_disto >= r)[0][0]
            self.centroids[cluster] = self.data[index]
            cluster += 1

    # find centroid with smallest angle with respect to data_point
    def __get_closest_centroid(self, data_point):
        min_distance = float("inf")
        closest = 0
        for index, centroid in self.centroids.items():
            dist = np.linalg.norm(np.asarray(data_point) - np.asarray(centroid))
            if dist < min_distance:
                min_distance = dist
                closest = index
        return closest

    # finds new centroids for each cluster
    def __rewrite_centroids(self):
        converged = True
        for i in range(0, self.k_clusters):
            # clustered data is of the form ([feature vector], label), group clusters by label and extract feature vector
            points_in_cluster = list(map(lambda x: x[0] ,list(filter(lambda x: x[1] == i, self.clustered_data))))
            new_centroid = [sum(i)/len(points_in_cluster) for i in zip(*points_in_cluster)]
            # if no centroids have changed, we can consider the algorithm converged
            if new_centroid != self.centroids[i]:
                converged = False
            self.centroids[i] = new_centroid
        return converged

    # returns an array "a" such that a[k] is the score for the dataset with k clusters, higher score is a better clustering job
    @staticmethod
    def recommend_k_clusters(dataset, max_clusters):
        # calculates how compact a cluster is, lower number is more compact
        def compactness(model):
            clusters = {}
            for centroid in model.centroids:
                clusters[centroid] = list(map(lambda x: x[0],list(filter(lambda x: x[1] == centroid, model.clustered_data))))
            total_compactedness = 0
            for index, cluster in clusters.items():
                # calculate pairwise distance between all points
                pairwise_distances = 0
                for x in range(0, len(cluster) - 1):
                    for y in range(x+1, len(cluster)):
                        pairwise_distances += np.linalg.norm(np.asarray(cluster[x]) - np.asarray(cluster[y]))
                total_compactedness += pairwise_distances/(2 * len(cluster))
            return total_compactedness
        # creates uniform random data with the same bounding box as the dataset
        def __bounding_box(dataset):
            X = dataset
            dimensions = len(X[0])
            mins = []
            maxs = []
            # create mins and maxs for each dimension
            for d in range(dimensions):
                mins.insert(d, min(X,key=lambda a:a[d])[d])
                maxs.insert(d, max(X,key=lambda a:a[d])[d])
            uniform_data = []
            #iterate all points in dataset
            for n in range(len(X)):
                    randomly_placed_vector = []
                    # for each point, iterate each dimension and place a random coordinate there
                    for k in range(dimensions):
                        randomly_placed_vector.append(random.uniform(mins[k], maxs[k]))
                    uniform_data.append(randomly_placed_vector)
            return uniform_data
        score = []
        score.insert(0, -100)
        score.insert(1, -100)
        for n in range(2, max_clusters+1):
            model = KMeans(k_clusters=n)
            model.fit(dataset)
            reference_model_compactness = 0
            B = 100 # possibly allow users to change this value
            # get average compactness of
            for i in range(1, B):
                reference_model = KMeans(k_clusters=n)
                reference_model.fit(__bounding_box(dataset))
                reference_model_compactness += compactness(reference_model)/B
            gap = math.log(reference_model_compactness) - math.log(compactness(model))
            print(math.log(reference_model_compactness), ' vs ' , math.log(compactness(model)))
            score.insert(n, gap)
        return score

    # returns an integer k representing the optimal amount of k_clusters for the dataset
    @staticmethod
    def optimal_k(dataset, max_clusters):
        clustering_scores = KMeans.recommend_k_clusters(dataset, max_clusters)
        return clustering_scores.index(max(clustering_scores))
