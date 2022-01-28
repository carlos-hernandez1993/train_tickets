# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt



#trenes = pd.read_csv("Alumno_Folder/Alumno_Curso/datascience_thebridge_9_21/Proyecto ML/SRC/Data/RAW/viajes-en-tren-sample.csv")

def preprocesado(df):

    #QUITAR TILDES CIUDADES DE ORIGEN
    df['origin'] = df['origin'].str.replace('CÓRDOBA','CORDOBA')
    df['origin'] = df['origin'].str.replace('LEÓN','LEON')
    df['origin'] = df['origin'].str.replace('MÁLAGA','MALAGA')


    #QUITAR TILDES CIUDADES DE DESTINO

    df['destination'] = df['destination'].str.replace('CÓRDOBA','CORDOBA')
    df['destination'] = df['destination'].str.replace('LEÓN','LEON')
    df['destination'] = df['destination'].str.replace('MÁLAGA','MALAGA')


    #REDUCIR DE TIPOS DE TRENES
    df['vehicle_type'] = df['vehicle_type'].str.replace('LD-AVE','AVE')
    df['vehicle_type'] = df['vehicle_type'].str.replace('AVE-LD','AVE')
    df['vehicle_type'] = df['vehicle_type'].str.replace('MD-AVE','AVE')
    df['vehicle_type'] = df['vehicle_type'].str.replace('AVE-MD','AVE')
    df['vehicle_type'] = df['vehicle_type'].str.replace('REGIONAL','REG.EXP')
    df['vehicle_type'] = df['vehicle_type'].str.replace('MD-LD','MD')
    df['vehicle_type'] = df['vehicle_type'].str.replace('LD-MD','LD')
    df['vehicle_type'] = df['vehicle_type'].str.replace('LD-AVANT','AVANT')
    df['vehicle_type'] = df['vehicle_type'].str.replace('REGIONAL.','REG.EXP')
    df['vehicle_type'] = df['vehicle_type'].str.replace('Intercity','INTERCITY')
    df['vehicle_type'] = df['vehicle_type'].str.replace('REG.EXP.','REG.EXP')


    #REDUCIR TIPOS DE VAGON
    df['vehicle_class'] = df['vehicle_class'].str.replace('Turista - Turista Plus','Turista')
    df['vehicle_class'] = df['vehicle_class'].str.replace('Turista Plus - Turista','Turista Plus')

    
   



    le = preprocessing.LabelEncoder()

    #TRANSFORMAR CIUDADES DE ORIGEN
    le.fit(df['origin'])
    df['origin_le'] = le.transform(df['origin'])

    #TRANSFORMAR CIUDADES DE DESTINO
    le.fit(df['destination'])
    df['destination_le'] = le.transform(df['destination'])

    #TRANSFORMAR TIPOS DE TRENES
    le.fit(df['vehicle_type'])
    df['vehicle_type_le'] = le.transform(df['vehicle_type'])

    #TRANSFORMAR TIPOS DE VAGON
    le.fit(df['vehicle_class'])
    df['vehicle_class_le'] = le.transform(df['vehicle_class'])

    #TRANSFORMAR TARIFAS
    le.fit(df['fare'])
    df['fare_le'] = le.transform(df['fare'])





    df['departure'] = pd.to_datetime(df['departure'])
    df['Month'] = df['departure'].dt.month
    df['Day'] = df['departure'].dt.day
    df['Day of the week'] = df['departure'].dt.dayofweek
    df['Hour'] = df['departure'].dt.hour

    df.drop(['company', 'origin', 'destination', 'departure', 'arrival', 'vehicle_type', 'vehicle_class', 'fare', 'meta', 'insert_date'], axis=1, inplace=True)

    return df



