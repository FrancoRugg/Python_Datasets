from sklearn.datasets import load_iris;
from sklearn.neighbors import KNeighborsClassifier; #Clase de modelo,el dato que no se encuentre etiquetado lo busca en los más parecidos
from sklearn.metrics import accuracy_score; #Score, que tan bien le fué en el entrenamiento 1 = Bueno, 0 = Malísimo
from sklearn.model_selection import train_test_split,cross_val_score,LeaveOneOut; #Entorno de prueba, para dividir datos de entrenamiento y de validaciones para entrenar al modelo.
import numpy as np;

iris = load_iris();
x = iris.data; #UNTAGGED
y = iris.target; #TAGGED

#x1,x2,y1,y2 = train_test_split(x,y,random_state=0,train_size=0.5); #SALTEAR

# print(x.shape);
# print(y.shape);
print(iris);

model = KNeighborsClassifier(n_neighbors=25); #Para que traiga un dato solo.

#model.fit(x1,y1); #Le paso los datos por separado, Matriz y Vector #SALTEAR

# y_predict = model.predict(x2); #SALTEAR
# print(accuracy_score(y2,y_predict)); #SALTEAR
loo = LeaveOneOut();
loo.get_n_splits(x);
print(cross_val_score(model,x,y,cv=loo)); #Con LeaveOneOut, ejecuta todos menos 1.


av = np.average(cross_val_score(model,x,y,cv=loo));
print(av); # Trae el promedio;









# print(model.predict(np.array([5.8,3.,1.5,9.6]).reshape(1,4)));



