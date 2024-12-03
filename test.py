import numpy as np;
from sklearn.cluster import KMeans; #Modelo K MEDIAS

from sklearn.datasets import make_moons

x, y = make_moons(n_samples=300, noise=0.1)

moons = make_moons();
print();


# for cluster_id in range(3):
#     indices = np.where(y_predict == cluster_id)[0]
    
#     true_ESPECIE = y[indices]
    
#     counts = Counter(true_ESPECIE)
    
#     # print(f"Cluster {cluster_id}:")
#     for specie_id, count in counts.items():
#         print(f"{ESPECIE[specie_id]}: {count} muestras")

# centroids = model.cluster_centers_
# print("\nCentroides de los clusters:")
# print(centroids)

