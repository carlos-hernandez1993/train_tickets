import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from Utils.preprocesado1 import preprocesado as prep1
import sys, os

import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def get_root_path(n):
    '''
    Esta función nos permite iterar sobre carpetas para añadir el path de nuestra carpeta raíz
    Argumentos:
        - n (int): el número de veces que iteraremos para llegar a la carpeta deseada
    '''
    path = os.getcwd() # para notebook ||| __file__ --> para .py
    for i in range(n):
        path = os.path.dirname(os.path.abspath(__file__))
        print('---------------')
        print(path)
        print('---------------')
    sys.path.append(path)

get_root_path(n=1)
print()
sys.path

path = pd.read_csv("Alumno_Folder/Alumno_Curso/datascience_thebridge_9_21/Proyecto ML/SRC/Data/RAW/viajes-en-tren-sample.csv")


DF_modificado = prep1(path);

X = DF_modificado[['duration','seats','origin_le','destination_le','vehicle_type_le','vehicle_class_le','fare_le','Month','Day','Day of the week','Hour']]
y = DF_modificado["price"]

model_RF = RandomForestRegressor(n_estimators= 50)
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.2)
model_RF.fit(X_train, y_train)

model_RF.predict(X_test)

y_pred = model_RF.predict(X_test)

print("MSE", mean_squared_error(y_test, y_pred))
print("RMSE", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE", mean_absolute_error(y_test, y_pred))
print('R2 score', r2_score(y_test, y_pred))


filename = 'trenes_Randomforest_model'

with open(filename, 'wb') as archivo_salida:
    pickle.dump(model_RF, archivo_salida)