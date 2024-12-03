import matplotlib.pyplot as plt;
from sklearn.naive_bayes import MultinomialNB; #EL MULTINOMIAL DISTRIBUYE DIFERENTE LOS DATOS.
from sklearn.feature_extraction.text import TfidfVectorizer;#TIME FREQUENCY INVERSE ... FREQUENCY VECTORIZER(SE FIJA QUE TANTO SE REPITE A NIVEL VECTORAL y )
from sklearn.model_selection import train_test_split;
from sklearn.metrics import classification_report; #PARECIDO AL ACCURACY_SCORE
import numpy as np;

import pandas as pd;

data = pd.read_json('News_Category_Dataset_v3.json',lines=True);

# print(data.head()); #ACCEDER AL HEAD PARA VER LOS DATOS.

#CONTIENE 5 filas, 6 columnas.

#Combinar titulo y descripción corta.

data['text'] = data['headline'] + " " + data['short_description'];

relevant_categories = ['WORLD NEWS','ENVIROMENT','PARENTING'];

#Se trae solo las categorías relevantes previamente ingresadas.
data = data[data['category'].isin(relevant_categories)];

# print(data.head());
# print(data['text']);

# AHORA PASA A TENER 7 COLUMNAS POR AGREGAR LA COLUMNA TEXT 

category_mapping = {
    'WORLD NEWS':0, #HUMAN RIGHTS
    'ENVIROMENT':1, #SUSTANTIBILITY
    'PARENTING':2   #CHILD ADVOCACY
    }

data['label'] = data['category'].apply(lambda x: category_mapping[x]);

# AGREGA COLUMNA LABEL

X = data['text'];
y = data['label'];

X_train,x_test,y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42);

vectorizer = TfidfVectorizer(stop_words='english',max_features=10000); #TE PIDE LAS 10000 PALABRAS QUE MÄS SE REPITAN EN EL TEXTO.

#CALCULA EL PUNTAJE POR CADA UNA DE LAS PALABRAS QUE SE REPITEN.
x_train_vect = vectorizer.fit_transform(X_train); #PUNTAJES
x_test_vect = vectorizer.transform(x_test);

# print(x_test_vect.head()); #ESTO NO SE PUEDE IMPRIMIR YA QUE SE CONVIRIÖ EN UNA MATRIZ.

model = MultinomialNB();

model.fit(x_train_vect,y_train); #Y_trains CATEGORIA DEL TEXTO.

y_pred = model.predict(x_test_vect);

print(classification_report(y_test,y_pred, target_names=category_mapping.keys())); #PARA SABER A QUE CATEGORIA PERTENECIAN LE HACEMOS EL KEYS AL CATEGORY_MAPPING.

myNew = pd.array(['How to be a good parent. Thebaby should not go to into the washer.'])

x_new = vectorizer.fit_transform(myNew);

print(model.predict(x_new));#PREDICE ETIQUETA PARA EL TEXTO INGRESADO.
# [2]

myNew = pd.array(['Biden is launching an attack to Donald Trump on Russia.'])

x_new = vectorizer.fit_transform(myNew);

print(model.predict(x_new));#PREDICE ETIQUETA PARA EL TEXTO INGRESADO.

