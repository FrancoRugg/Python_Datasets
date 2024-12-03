import pandas as pd;
import numpy as np;
from sklearn import preprocessing;
from sklearn.model_selection import train_test_split,cross_val_score,LeaveOneOut,GridSearchCV;
from sklearn.tree import DecisionTreeClassifier,plot_tree;
from sklearn.metrics import accuracy_score;# Mide si el modelo entrena bien
from matplotlib import pyplot as plt; #Para que te aparezca el arbol. 

def encode_features(df, features = ['embarked','cabin','sex']): #Parametrizamos las features.
    # features = ['embarked','cabin','sex'];
    for feature in features:
        le= preprocessing.LabelEncoder();#Genera un etiquetor - le(label encoder);
        le=le.fit(df[feature]);#Entrena cada columna.
        df[feature]=le.transform(df[feature]);#Reemplaza a columna de texto por numerica.
        
    return df;
        
def fillNulls(df):
    #Lenar los valores nulos
    df['age'].fillna({'age':df['age'].mean()},inplace=True); #Los llena en base al promedio de las edades.
    df['cabin'].fillna({'cabin':'N'},inplace=True); #Los llena en base al promedio de las edades.
    df['embarked'].fillna({'embarked':'N'},inplace=True); #Los llena en base al promedio de las edades.
    # inplace=True, Todas las vistas van a tener el mismo valor nuevo.
    return df;
def drop_features(df,features=['name','ticket']):
    df.drop(features,axis=1,inplace=True);#Axis = 1, elimina la columna entera.
    return df;

def format_features(df):
    df['cabin']=df['cabin'].str[0:1];#Se trae desde el primer elemento, quedandose con una posición.
    # print('Cabin dist', df['cabin'].value_counts()); #Retorna una serie de valores.
    df = encode_features(df);
    return df;
    
#Toma datos de la librería administrada.
titanic_df = pd.read_csv('train.csv');
#LLena los Nulls.
titanic_df = fillNulls(titanic_df);

#Mejora distribución de cabins.
titanic_df = format_features(titanic_df);

#Elimina columnas innecesarias.
titanic_df = drop_features(titanic_df);

# print(titanic_df.head());

y_titanic_df = titanic_df['survived'];
x_titanic_df = titanic_df.drop('survived',axis=1);

#test_size=0.2 20% de los datos para entrenar y el 80% para probar.
#random_state=11 Semilla, buena semilla es núm par o primo.
X_train,X_test,y_train,y_test = train_test_split(x_titanic_df,y_titanic_df,test_size=0.2,random_state=11)#Mayúscula es lo generado, minúscula es lo real.

model = DecisionTreeClassifier(random_state=11);#Se le pasa el orden en el que quiero minar los datos.

#Parametros para tener un mejor resultado
parameters = {'max_depth':[2,3,5,10],'min_samples_split':[2,3,5],'min_samples_leaf':[1,5,8]};

gridModel = GridSearchCV(model,param_grid=parameters,scoring='accuracy',cv=5); #Cv = 5, hace validación cruzada 5 veces.

gridModel.fit(X_train,y_train);

print('Best parameter combination is: ', gridModel.best_params_);
print('Best accurancy is: ', gridModel.best_score_);

# Y_predict= model.predict(X_test); # Predice etiquetas de validación
# print("Accuracy score:", accuracy_score(y_test,Y_predict));

newModel = gridModel.best_estimator_;
newModel.fit(X_train,y_train);

# plt.figure();
plot_tree(newModel); # Te devuelve que tipo de semilla usó.
plt.show();
# plt.savefig('tree.svg',format='svg',bbox_inches='tight');# Te genera el dato del arbol.




