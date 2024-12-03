import numpy as np;
from sklearn.cluster import KMeans; #Modelo K MEDIAS

X = np.array([25,38,47,51,10,77,89,91,52,1,55,12,8,9,63]) #Generar array;
X = X.reshape(15,1);# PARA DECIR DE CUANTO VAN A SER LAS FILAS Y COLUMNAS
Xnew = np.array([33,79,28]).reshape(3,1)

# n_ckusters CANTIDAD DE COLUMNAS
# random_state SEMILLA DE MEZCLA
model = KMeans(n_clusters = 3,random_state=11); #DATOS DEL MODELO
model.fit(X);# ENTRENO EL MODELO
y_predict = model.predict(X);# AVERIGUAR VALORES PARA CADA NÜMERO
ynew = model.predict(Xnew)
# 
# PREGUNTAR ACÄ LO QUE IBA EN Y
# 
i = 0;
cero = np.array([]);
one = np.array([]);
two = np.array([]);

while i < 15:
    if y_predict[i] == 0:
        cero = np.append(cero, X[i])
    elif y_predict[i] == 1:
        one = np.append(one, X[i])
    elif y_predict[i] == 2:
        two = np.append(two, X[i])
    print(X[i],"-----",y_predict[i]);
    i += 1;
print(cero);
print(one);
print(two);

print(ynew);

# Calcular el promedio de cada grupo
avg_cero = np.mean(cero,axis=0); 
avg_one = np.mean(one,axis=0);
avg_two = np.mean(two,axis=0);

# Mostrar los promedios
print(f"Promedio de 'cero': {avg_cero}")
print(f"Promedio de 'one': {avg_one}")
print(f"Promedio de 'two': {avg_two}")
# print(X);
# print(y_predict);