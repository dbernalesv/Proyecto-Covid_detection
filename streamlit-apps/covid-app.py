# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 23:09:10 2021

@author: diego
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle


st.write("""
# Aplicación para predecir el covid
Está aplicación ayuda a predecir si eres covid positivo o negativo según algunos síntomas
""")

st.sidebar.header('User Input Features/ Variables de entrada del usuario')

st.sidebar.markdown("""
Selecciona Si o No , para saber que sítomas tienes y si eres mayor de 60 años
Luego Selecciona tu tipo de contacto
""")

# Colecta inputs del usuario en un dataframe/ Collects user input features into dataframe

def user_input_features():
    Tos = st.sidebar.selectbox('Tos',('Si','No'))
    fiebre = st.sidebar.selectbox('Fiebre',('Si','No'))
    dolor_de_garganta = st.sidebar.selectbox('Dolor de Garganta',('Si','No'))
    dificultad_para_respirar = st.sidebar.selectbox('Dificultad para respirar',('Si','No'))
    dolor_de_cabeza = st.sidebar.selectbox('Dolor de Cabeza',('Si','No'))
    mayor_a_60_años = st.sidebar.selectbox('Mayor a 60 años',('Si','No'))
    indicacion= st.sidebar.selectbox('Tipo de contacto',('He estado en el extranjero','Tuve contacto con un caso confirmado','Otro(buses,supermercados,etc)'))
    data = {'Tos': Tos,
                'fiebre': fiebre,
                'dolor_de_garganta': dolor_de_garganta,
                'dificultad_para_respirar': dificultad_para_respirar,
                'dolor_de_cabeza': dolor_de_cabeza,
                'mayor_a_60_años': mayor_a_60_años,
                'indicación_de_test':indicacion}
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

# Encoding of categorical features/ Codificando variables categóricas

column_names = ["Tos","fiebre","dolor_de_garganta","dificultad_para_respirar","dolor_de_cabeza","mayor_a_60_años"]
for col in column_names:
    num_to_cat = {'No':0, 'Si':1 }
    input_df.replace({col: num_to_cat},inplace=True)

num_to_cat1 = {'He estado en el extranjero':0, 'Tuve contacto con un caso confirmado':1,'Otro(buses,supermercados,etc)':2}
input_df.replace({'indicación_de_test': num_to_cat1},inplace=True)

# Displays the user input features
st.subheader('User Input features/Variables de entrada del usuario')



st.write('Sus datos: ')
st.write(input_df)

# Reads in saved classification model/ cargamos el modelo de clasificación guardado
load_clf = pickle.load(open('covid_clf.pkl', 'rb'))

# Apply model to make predictions / aplicando el modelo para hacer predicciones
prediction = load_clf.predict(input_df)
prediction_proba=load_clf.predict_proba(input_df)


st.subheader('Prediction/Predicción')
resultado = np.array(['Negativo/Negative','Positivo/Positive'])
st.write(resultado[prediction])

st.subheader('Prediction Probability/ Probabilidad de Predicción')
st.write(prediction_proba)