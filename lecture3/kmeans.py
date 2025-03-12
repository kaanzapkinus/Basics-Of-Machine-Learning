import matplotlib.pyplot as plt
#from kneed import KneeLocator
from matplotlib import cm
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

features, true_labels = make_blobs(
 n_samples=500,
 centers=5,
 cluster_std=1.5,
 random_state=1)

x = [x[0] for x in features]
y = [x[1] for x in features]

hsv_modified = cm.get_cmap('rainbow', 5)

plt.scatter(x, y, cmap=hsv_modified, c=true_labels)
plt.show()


hsv_modified = cm.get_cmap('rainbow', 5)
kmeans = KMeans(n_clusters=5, init="random", n_init=50, max_iter=1, random_state=2)
kmeans.fit(features)
cluster_predict = kmeans.fit_predict(features)
cluster_centers = kmeans.cluster_centers_
x2 = [x[0] for x in cluster_centers]
y2 = [x[1] for x in cluster_centers]
plt.scatter(x, y, cmap=hsv_modified, c=cluster_predict)
plt.scatter(x2, y2, marker='*', s = 500, c='k')
plt.show()

distortions = []
K = range(2, 10)
for k in K:
    kmeanModel = KMeans(n_clusters=k, init="k-means++", n_init=50, max_iter=500, random_state=2)
    kmeanModel.fit(features)
    distortions.append(kmeanModel.inertia_)


plt.plot(K, distortions, 'bx-')
plt.xlabel('number of clusters')
plt.ylabel('Distortion')
plt.title('Elbow method')
plt.show()


plt.scatter(x, y, cmap=hsv_modified, c=cluster_predict)
plt.show()
hsv_modified = cm.get_cmap('rainbow', 30)
kmeans = KMeans(n_clusters=5, init="random", n_init=50, max_iter=1, random_state=2)
kmeans.fit(features)
from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()
features = ms.fit_transform(features)

cluster_predict = kmeans.fit_predict(features)
cluster_centers = kmeans.cluster_centers_
x = [x[0] for x in features]
y = [x[1] for x in features]
plt.scatter(x, y, cmap=hsv_modified, c=cluster_predict)
plt.show()






