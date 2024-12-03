# VER EJEMPLO REAL EN LA DOCUMENTACIÖN 
# ESTÄ BUENO PARA CLASIFICAR TEXTO Y DEFINIR SI UN DATO ES SPAM O NO
from sklearn.datasets import make_blobs;
import matplotlib.pyplot as plt;
from sklearn.naive_bayes import GaussianNB;
import numpy as np;


X,y =make_blobs(100,2,centers=2,random_state=11,cluster_std=1.5);
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="RdBu");
# plt.show();

model = GaussianNB();
model.fit(X,y);

rango = np.random.RandomState(11);

# SE GENERAN 2000 PUNTOS AL AZAR DE PRUEBA
Xnew = [-6,-14] + [14,18] * rango.rand(2000,2);

#PREDICCIÖN
ynew = model.predict(Xnew);
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="RdBu");

#EJES DEL GRAFICO
lim = plt.axis();

#S ES EL TAMAÑO
#alpha ES LA OPACIDAD
plt.scatter(Xnew[:,0],Xnew[:,1],c=ynew,s=20,cmap="RdBu",alpha=0.2);
plt.axis(lim);

# plt.show();

#TE TRAE LAS PROBABILIDADES DE PERTENECER AL EQUIPO ROJO O AL AZUL
yprob = model.predict_proba(Xnew);
print(yprob[-15:].round(2));