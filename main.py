import matplotlib.pyplot as plt;
import numpy as np;
from sklearn.linear_model import LinearRegression;
# npm install scikit-learn
# npm install matplotlib

rgn = np.random.RandomState(42); #Setea como semilla el número ingresado.
x = 10 * rgn.rand(50); #Genera 50 números random a partir de la semilla y los multiplica x 10.
y = 2 * x - 1 + rgn.randn(50);
plt.scatter(x, y); #Genera el gráfico.
plt.show(); #Muestra el grafico.

model = LinearRegression(fit_intercept=True); #Entrena el modelo en base a los datos dados

X=x.reshape(x.size,1);

model.fit(X,y);

print(model.coef_, model.intercept_); #Genera un gráfico lineal con los datos

