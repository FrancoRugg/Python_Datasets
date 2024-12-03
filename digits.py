from sklearn.datasets import load_digits;
import matplotlib.pyplot as plt;
import numpy as np;
from sklearn.cluster import KMeans; #Modelo K MEDIAS
from scipy.stats import mode; # SE FIJA SI LA ETIQUETA ES IGUAL A LA VERDADERA; CASO CONTRARIO LA PERMUTA;
from sklearn.metrics import accuracy_score;

digits = load_digits(); # DATOS A UTILIZAR
print(digits.data[0]);
# RANDOM STATE INICIALIZARLO EN NUM PRIMO O IMPAR
model = KMeans(n_clusters=10,random_state=11);
clusters = model.fit_predict(digits.data);
print(model.cluster_centers_.shape);
print("clusters",clusters);
sample = np.random.rand(64);
sample = sample.reshape(1,64);

fig,ax = plt.subplots(2,5,figsize=(8,3));
centers = model.cluster_centers_.reshape(10,8,8);

# LA FUNCIÖN XIP TE ARMA PAREJAS
for axi,center in zip(ax.flat,centers):
    axi.set(xticks=[],yticks=[])
    axi.imshow(center,interpolation='nearest',cmap=plt.cm.binary)
# plt.show();

# zeros_like ARMA ARRAY CON TODOS 0 CON LA MISMA FORMA QUE DEVUELVE LABELS
labels = np.zeros_like(clusters);
for i in range(10):
    mask = (clusters == i);
    # mode SE FIJA SI LA ETIQUETA ES IGUAL A LA VERDADERA; CASO CONTRARIO LA PERMUTA;
    labels[mask] = mode(digits.target[mask])[0];

#IMPRIME LAS ETIQUETAS VERDADERAS VS LAS CALCULADAS
print(accuracy_score(digits.target,labels));
print(model.predict(sample));#PREDICE A DONDE ENVIAR ESTOS DATOS ALEATORIOS;

    


    


# fig = plt.figure(figsize=(6,6));
# fig.subplots_adjust(left = 0,right=1,bottom=0,top=1,hspace=0.05);

# for i in range(64):
#     ax = fig.add_subplot(8,8,i + 1,xticks=[],yticks=[]);
    