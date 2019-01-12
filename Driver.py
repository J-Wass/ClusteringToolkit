from ClusteringToolkit import KMeans
import matplotlib.pyplot as plt
import numpy as np
import random

def gen_sample_data(dims, nsamp):
    idx = np.random.choice(np.prod(dims), nsamp, replace=False)
    return np.vstack(np.unravel_index(idx, dims)).T

cluster1 = list(gen_sample_data((200,200), 10))
cluster2 = list(map(lambda x: x+300,gen_sample_data((200,50), 10)))
cluster3 = list(map(lambda x: x+600,gen_sample_data((100,150), 10)))
cluster4 = list(map(lambda x: x+900,gen_sample_data((200,100), 10)))
data = np.asarray(cluster1+cluster2+cluster3+cluster4)
k = KMeans.optimal_k(dataset=data.tolist(), max_clusters=8, algorithm="silhouette")
print('recommended clusters (silhouette): ' , k)
k = KMeans.optimal_k(dataset=data.tolist(), max_clusters=8, algorithm="gap")
print('recommended clusters (gap): ' , k)
model = KMeans(k_clusters = k)
labels = model.fit(data.tolist())
labelled_data = list(zip(data, labels))
colors = ["red", "blue", "green", "yellow", "orange", "cyan", "black", "magenta", "#1abc9c", " #abb2b9"]
for i in range (0,k):
    x,y = zip(*list(map(lambda y: y[0], list(filter(lambda x: x[1] == i, labelled_data)))))
    plt.scatter(x,y, c = colors[i])

plt.show()
