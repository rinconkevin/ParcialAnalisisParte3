#Interfaz
from tkinter import ttk, Tk, Frame,Button 
import tkinter as tk

#Graficas
import pandas as pd
import matplotlib.pyplot as plot
from  scipy import stats
import numpy  as np
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,  NavigationToolbar2Tk) 
#from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math
from math import log
import statistics
import random


main_window = tk.Tk()

url= "wine - copia.csv"
datos = pd.read_csv(url)

columnas = ['Tipo', 'alcohol', 'acido malico','Ceniza','alcalinidad de la ceniza','magnesio', "fenoles totales", "flavonoides", "fenoles no flavonoides", "proantocianinas", "fntensidad de color", "tonalidad", "vinos diluidos", "prolina"]
datos.columns = columnas

#---------------------------- Modelo-------------------------------------------#

def Tabla2(Entrada, Entrada2, df, repeticiones): 
    n = 3
    m = 3

    matriz = []

    for i in range(n):
        matriz.append([])
        for j in range(m):
            matriz[i].append(0)

    print(matriz)
    for i in range (repeticiones):
        print("----------------------------------------------------------------------------")
        print(type(df))
        print(df.shape)
        pruebita2 = df.describe()
        labels = dict(zip(df.Tipo.unique(), df.Tipo.unique()))
        print(df['Tipo'].value_counts())

        #taking random record and storing in xq
        xq = df.sample()

        # droping the xq from data using index value
        df.drop(xq.index, inplace=True)
        print(df.shape)
        xq_final = pd.DataFrame(xq[[Entrada, Entrada2]])
        final = xq_final

        # calculating ecludian distance
        def cal_distance(x):      
            a = x.to_numpy()
            b = xq_final.to_numpy()    
            distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(a, b[0])]))
            return distance

        # calculating distance
        df['distance'] = df[[Entrada, Entrada2]].apply(cal_distance, axis=1)

        #sorting the values based on distance
        df_sort = df.sort_values('distance',ascending=True)

        # taking top 11 records because k is 11
        df_after_sort = df_sort.head(11)

        pruebita3 = df_after_sort.reset_index()

        print(df_after_sort.iloc[0])

        #q esta pasando aki?
        count = [0 for i in range(0, len(df['Tipo'].unique()))]
        for xi in range(0, len(df_after_sort)):       
            if df_after_sort.iloc[xi]['Tipo'] == 1:        
                count[0] = count[0]+1
            elif df_after_sort.iloc[xi]['Tipo'] == 2:        
                count[1] = count[1]+1
            elif df_after_sort.iloc[xi]['Tipo'] == 3:        
                count[2] = count[2]+1
                
        def max_num_in_list_label(list):
            maxpos = list.index(max(list)) +1
            return labels[maxpos]
        
        #getting the label and verifying with the class label in xq
        if max_num_in_list_label(count) in xq.values:
            xqValues = xq.values
            xqValues2 = max_num_in_list_label(count)
            print("Prueba Positiva")
            print(xqValues[0][0])
            matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] = matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] + 1
        else:
            xqValues = xq.values
            print(xqValues)
            xqValues2 = max_num_in_list_label(count)
            print(xqValues2)
            print("Prueva Negativa")
            print(xqValues[0][0])
            matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] = matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] + 1
        print("----------------------------------------------------------------------------")

    Presicion = []
    Sensibilidad = []

    sumaFilas =  [sum(i) for i in matriz]
    sumaColumnas =  [sum(i) for i in zip(*matriz)]
    print("Matriz")
    print(matriz[0])
    print(matriz[1])
    print(matriz[2])
    print("Sumatoria de Columnas")      
    print(sumaColumnas)
    print("Sumatoria de Filas")
    print(sumaFilas)

    print("Presicion")
    for i in range(n):
        for j in range(m):
            if(j == i):
                if(matriz[i][j] == 0 and  sumaFilas[i] == 0):
                    Presicion.append("Infinito")
                else:
                    Presicion.append((matriz[i][j]/sumaFilas[i])*100)
    print(Presicion)
    print("Sensibilidad")
    for i in range(n):
        for j in range(m):
            if(j == i):
                if(matriz[i][j] == 0 and  sumaFilas[i] == 0):
                    Sensibilidad.append("Infinito")
                else:
                    Sensibilidad.append((matriz[i][j]/sumaColumnas[i])*100)
    print(Sensibilidad)      
            
def Tabla3(Entrada, Entrada2,Entrada3, df, repeticiones): 
    n = 3
    m = 3

    matriz = []

    for i in range(n):
        matriz.append([])
        for j in range(m):
            matriz[i].append(0)

    print(matriz)
    for i in range (repeticiones):
        print("----------------------------------------------------------------------------")
        print(type(df))
        print(df.shape)
        pruebita2 = df.describe()
        labels = dict(zip(df.Tipo.unique(), df.Tipo.unique()))
        print(df['Tipo'].value_counts())

        #taking random record and storing in xq
        xq = df.sample()

        # droping the xq from data using index value
        df.drop(xq.index, inplace=True)
        print(df.shape)
        xq_final = pd.DataFrame(xq[[Entrada, Entrada2, Entrada3]])
        final = xq_final

        # calculating ecludian distance
        def cal_distance(x):      
            a = x.to_numpy()
            b = xq_final.to_numpy()    
            distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(a, b[0])]))
            return distance

        # calculating distance
        df['distance'] = df[[Entrada, Entrada2, Entrada3]].apply(cal_distance, axis=1)

        #sorting the values based on distance
        df_sort = df.sort_values('distance',ascending=True)

        # taking top 11 records because k is 11
        df_after_sort = df_sort.head(11)

        pruebita3 = df_after_sort.reset_index()

        print(df_after_sort.iloc[0])

        #q esta pasando aki?
        count = [0 for i in range(0, len(df['Tipo'].unique()))]
        for xi in range(0, len(df_after_sort)):       
            if df_after_sort.iloc[xi]['Tipo'] == 1:        
                count[0] = count[0]+1
            elif df_after_sort.iloc[xi]['Tipo'] == 2:        
                count[1] = count[1]+1
            elif df_after_sort.iloc[xi]['Tipo'] == 3:        
                count[2] = count[2]+1
                
        def max_num_in_list_label(list):
            maxpos = list.index(max(list)) +1
            return labels[maxpos]
        
        #getting the label and verifying with the class label in xq
        if max_num_in_list_label(count) in xq.values:
            xqValues = xq.values
            xqValues2 = max_num_in_list_label(count)
            print("Prueba Positiva")
            print(xqValues[0][0])
            matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] = matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] + 1
        else:
            xqValues = xq.values
            print(xqValues)
            xqValues2 = max_num_in_list_label(count)
            print(xqValues2)
            print("Prueva Negativa")
            print(xqValues[0][0])
            matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] = matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] + 1
        print("----------------------------------------------------------------------------")

    Presicion = []
    Sensibilidad = []

    sumaFilas =  [sum(i) for i in matriz]
    sumaColumnas =  [sum(i) for i in zip(*matriz)]
    print("Matriz")
    print(matriz[0])
    print(matriz[1])
    print(matriz[2])
    print("Sumatoria de Columnas")      
    print(sumaColumnas)
    print("Sumatoria de Filas")
    print(sumaFilas)

    print("Presicion")
    for i in range(n):
        for j in range(m):
            if(j == i):
                if(matriz[i][j] == 0 and  sumaFilas[i] == 0):
                    Presicion.append("Infinito")
                else:
                    Presicion.append((matriz[i][j]/sumaFilas[i])*100)
    print(Presicion)
    print("Sensibilidad")
    for i in range(n):
        for j in range(m):
            if(j == i):
                if(matriz[i][j] == 0 and  sumaFilas[i] == 0):
                    Sensibilidad.append("Infinito")
                else:
                    Sensibilidad.append((matriz[i][j]/sumaColumnas[i])*100)
    print(Sensibilidad)      

def Tabla4(Entrada, Entrada2,Entrada3,Entrada4, df, repeticiones): 
    n = 3
    m = 3

    matriz = []

    for i in range(n):
        matriz.append([])
        for j in range(m):
            matriz[i].append(0)

    print(matriz)
    for i in range (repeticiones):
        print("----------------------------------------------------------------------------")
        print(type(df))
        print(df.shape)
        pruebita2 = df.describe()
        labels = dict(zip(df.Tipo.unique(), df.Tipo.unique()))
        print(df['Tipo'].value_counts())

        #taking random record and storing in xq
        xq = df.sample()

        # droping the xq from data using index value
        df.drop(xq.index, inplace=True)
        print(df.shape)
        xq_final = pd.DataFrame(xq[[Entrada, Entrada2, Entrada3, Entrada4]])
        final = xq_final

        # calculating ecludian distance
        def cal_distance(x):      
            a = x.to_numpy()
            b = xq_final.to_numpy()    
            distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(a, b[0])]))
            return distance

        # calculating distance
        df['distance'] = df[[Entrada, Entrada2, Entrada3, Entrada4]].apply(cal_distance, axis=1)

        #sorting the values based on distance
        df_sort = df.sort_values('distance',ascending=True)

        # taking top 11 records because k is 11
        df_after_sort = df_sort.head(11)

        pruebita3 = df_after_sort.reset_index()

        print(df_after_sort.iloc[0])

        #q esta pasando aki?
        count = [0 for i in range(0, len(df['Tipo'].unique()))]
        for xi in range(0, len(df_after_sort)):       
            if df_after_sort.iloc[xi]['Tipo'] == 1:        
                count[0] = count[0]+1
            elif df_after_sort.iloc[xi]['Tipo'] == 2:        
                count[1] = count[1]+1
            elif df_after_sort.iloc[xi]['Tipo'] == 3:        
                count[2] = count[2]+1
                
        def max_num_in_list_label(list):
            maxpos = list.index(max(list)) +1
            return labels[maxpos]
        
        #getting the label and verifying with the class label in xq
        if max_num_in_list_label(count) in xq.values:
            xqValues = xq.values
            xqValues2 = max_num_in_list_label(count)
            print("Prueba Positiva")
            print(xqValues[0][0])
            matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] = matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] + 1
        else:
            xqValues = xq.values
            print(xqValues)
            xqValues2 = max_num_in_list_label(count)
            print(xqValues2)
            print("Prueva Negativa")
            print(xqValues[0][0])
            matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] = matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] + 1
        print("----------------------------------------------------------------------------")

    Presicion = []
    Sensibilidad = []

    sumaFilas =  [sum(i) for i in matriz]
    sumaColumnas =  [sum(i) for i in zip(*matriz)]
    print("Matriz")
    print(matriz[0])
    print(matriz[1])
    print(matriz[2])
    print("Sumatoria de Columnas")      
    print(sumaColumnas)
    print("Sumatoria de Filas")
    print(sumaFilas)

    print("Presicion")
    for i in range(n):
        for j in range(m):
            if(j == i):
                if(matriz[i][j] == 0 and  sumaFilas[i] == 0):
                    Presicion.append("Infinito")
                else:
                    Presicion.append((matriz[i][j]/sumaFilas[i])*100)
    print(Presicion)
    print("Sensibilidad")
    for i in range(n):
        for j in range(m):
            if(j == i):
                if(matriz[i][j] == 0 and  sumaFilas[i] == 0):
                    Sensibilidad.append("Infinito")
                else:
                    Sensibilidad.append((matriz[i][j]/sumaColumnas[i])*100)
    print(Sensibilidad)     

def Tabla5(Entrada, Entrada2,Entrada3,Entrada4,Entrada5, df, repeticiones): 
    n = 3
    m = 3

    matriz = []

    for i in range(n):
        matriz.append([])
        for j in range(m):
            matriz[i].append(0)

    print(matriz)
    for i in range (repeticiones):
        print("----------------------------------------------------------------------------")
        print(type(df))
        print(df.shape)
        pruebita2 = df.describe()
        labels = dict(zip(df.Tipo.unique(), df.Tipo.unique()))
        print(df['Tipo'].value_counts())

        #taking random record and storing in xq
        xq = df.sample()

        # droping the xq from data using index value
        df.drop(xq.index, inplace=True)
        print(df.shape)
        xq_final = pd.DataFrame(xq[[Entrada, Entrada2, Entrada3, Entrada4, Entrada5]])
        final = xq_final

        # calculating ecludian distance
        def cal_distance(x):      
            a = x.to_numpy()
            b = xq_final.to_numpy()    
            distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(a, b[0])]))
            return distance

        # calculating distance
        df['distance'] = df[[Entrada, Entrada2, Entrada3, Entrada4, Entrada5]].apply(cal_distance, axis=1)

        #sorting the values based on distance
        df_sort = df.sort_values('distance',ascending=True)

        # taking top 11 records because k is 11
        df_after_sort = df_sort.head(11)

        pruebita3 = df_after_sort.reset_index()

        print(df_after_sort.iloc[0])

        #q esta pasando aki?
        count = [0 for i in range(0, len(df['Tipo'].unique()))]
        for xi in range(0, len(df_after_sort)):       
            if df_after_sort.iloc[xi]['Tipo'] == 1:        
                count[0] = count[0]+1
            elif df_after_sort.iloc[xi]['Tipo'] == 2:        
                count[1] = count[1]+1
            elif df_after_sort.iloc[xi]['Tipo'] == 3:        
                count[2] = count[2]+1
                
        def max_num_in_list_label(list):
            maxpos = list.index(max(list)) +1
            return labels[maxpos]
        
        #getting the label and verifying with the class label in xq
        if max_num_in_list_label(count) in xq.values:
            xqValues = xq.values
            xqValues2 = max_num_in_list_label(count)
            print("Prueba Positiva")
            print(xqValues[0][0])
            matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] = matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] + 1
        else:
            xqValues = xq.values
            print(xqValues)
            xqValues2 = max_num_in_list_label(count)
            print(xqValues2)
            print("Prueva Negativa")
            print(xqValues[0][0])
            matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] = matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] + 1
        print("----------------------------------------------------------------------------")

    Presicion = []
    Sensibilidad = []

    sumaFilas =  [sum(i) for i in matriz]
    sumaColumnas =  [sum(i) for i in zip(*matriz)]
    print("Matriz")
    print(matriz[0])
    print(matriz[1])
    print(matriz[2])
    print("Sumatoria de Columnas")      
    print(sumaColumnas)
    print("Sumatoria de Filas")
    print(sumaFilas)

    print("Presicion")
    for i in range(n):
        for j in range(m):
            if(j == i):
                if(matriz[i][j] == 0 and  sumaFilas[i] == 0):
                    Presicion.append("Infinito")
                else:
                    Presicion.append((matriz[i][j]/sumaFilas[i])*100)
    print(Presicion)
    print("Sensibilidad")
    for i in range(n):
        for j in range(m):
            if(j == i):
                if(matriz[i][j] == 0 and  sumaFilas[i] == 0):
                    Sensibilidad.append("Infinito")
                else:
                    Sensibilidad.append((matriz[i][j]/sumaColumnas[i])*100)
    print(Sensibilidad)  

def Tabla6(Entrada, Entrada2,Entrada3,Entrada4,Entrada5,Entrada6, df, repeticiones): 
    n = 3
    m = 3

    matriz = []

    for i in range(n):
        matriz.append([])
        for j in range(m):
            matriz[i].append(0)

    print(matriz)
    for i in range (repeticiones):
        print("----------------------------------------------------------------------------")
        print(type(df))
        print(df.shape)
        pruebita2 = df.describe()
        labels = dict(zip(df.Tipo.unique(), df.Tipo.unique()))
        print(df['Tipo'].value_counts())

        #taking random record and storing in xq
        xq = df.sample()

        # droping the xq from data using index value
        df.drop(xq.index, inplace=True)
        print(df.shape)
        xq_final = pd.DataFrame(xq[[Entrada, Entrada2, Entrada3, Entrada4, Entrada5, Entrada6]])
        final = xq_final

        # calculating ecludian distance
        def cal_distance(x):      
            a = x.to_numpy()
            b = xq_final.to_numpy()    
            distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(a, b[0])]))
            return distance

        # calculating distance
        df['distance'] = df[[Entrada, Entrada2, Entrada3, Entrada4, Entrada5, Entrada6]].apply(cal_distance, axis=1)

        #sorting the values based on distance
        df_sort = df.sort_values('distance',ascending=True)

        # taking top 11 records because k is 11
        df_after_sort = df_sort.head(11)

        pruebita3 = df_after_sort.reset_index()

        print(df_after_sort.iloc[0])

        #q esta pasando aki?
        count = [0 for i in range(0, len(df['Tipo'].unique()))]
        for xi in range(0, len(df_after_sort)):       
            if df_after_sort.iloc[xi]['Tipo'] == 1:        
                count[0] = count[0]+1
            elif df_after_sort.iloc[xi]['Tipo'] == 2:        
                count[1] = count[1]+1
            elif df_after_sort.iloc[xi]['Tipo'] == 3:        
                count[2] = count[2]+1
                
        def max_num_in_list_label(list):
            maxpos = list.index(max(list)) +1
            return labels[maxpos]
        
        #getting the label and verifying with the class label in xq
        if max_num_in_list_label(count) in xq.values:
            xqValues = xq.values
            xqValues2 = max_num_in_list_label(count)
            print("Prueba Positiva")
            print(xqValues[0][0])
            matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] = matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] + 1
        else:
            xqValues = xq.values
            print(xqValues)
            xqValues2 = max_num_in_list_label(count)
            print(xqValues2)
            print("Prueva Negativa")
            print(xqValues[0][0])
            matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] = matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] + 1
        print("----------------------------------------------------------------------------")

    Presicion = []
    Sensibilidad = []

    sumaFilas =  [sum(i) for i in matriz]
    sumaColumnas =  [sum(i) for i in zip(*matriz)]
    print("Matriz")
    print(matriz[0])
    print(matriz[1])
    print(matriz[2])
    print("Sumatoria de Columnas")      
    print(sumaColumnas)
    print("Sumatoria de Filas")
    print(sumaFilas)

    print("Presicion")
    for i in range(n):
        for j in range(m):
            if(j == i):
                if(matriz[i][j] == 0 and  sumaFilas[i] == 0):
                    Presicion.append("Infinito")
                else:
                    Presicion.append((matriz[i][j]/sumaFilas[i])*100)
    print(Presicion)
    print("Sensibilidad")
    for i in range(n):
        for j in range(m):
            if(j == i):
                if(matriz[i][j] == 0 and  sumaFilas[i] == 0):
                    Sensibilidad.append("Infinito")
                else:
                    Sensibilidad.append((matriz[i][j]/sumaColumnas[i])*100)
    print(Sensibilidad) 

def Tabla7(Entrada, Entrada2,Entrada3,Entrada4,Entrada5,Entrada6,Entrada7, df, repeticiones): 
    n = 3
    m = 3

    matriz = []

    for i in range(n):
        matriz.append([])
        for j in range(m):
            matriz[i].append(0)

    print(matriz)
    for i in range (repeticiones):
        print("----------------------------------------------------------------------------")
        print(type(df))
        print(df.shape)
        pruebita2 = df.describe()
        labels = dict(zip(df.Tipo.unique(), df.Tipo.unique()))
        print(df['Tipo'].value_counts())

        #taking random record and storing in xq
        xq = df.sample()

        # droping the xq from data using index value
        df.drop(xq.index, inplace=True)
        print(df.shape)
        xq_final = pd.DataFrame(xq[[Entrada, Entrada2, Entrada3, Entrada4, Entrada5, Entrada6, Entrada7]])
        final = xq_final

        # calculating ecludian distance
        def cal_distance(x):      
            a = x.to_numpy()
            b = xq_final.to_numpy()    
            distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(a, b[0])]))
            return distance

        # calculating distance
        df['distance'] = df[[Entrada, Entrada2, Entrada3, Entrada4, Entrada5, Entrada6, Entrada7]].apply(cal_distance, axis=1)

        #sorting the values based on distance
        df_sort = df.sort_values('distance',ascending=True)

        # taking top 11 records because k is 11
        df_after_sort = df_sort.head(11)

        pruebita3 = df_after_sort.reset_index()

        print(df_after_sort.iloc[0])

        #q esta pasando aki?
        count = [0 for i in range(0, len(df['Tipo'].unique()))]
        for xi in range(0, len(df_after_sort)):       
            if df_after_sort.iloc[xi]['Tipo'] == 1:        
                count[0] = count[0]+1
            elif df_after_sort.iloc[xi]['Tipo'] == 2:        
                count[1] = count[1]+1
            elif df_after_sort.iloc[xi]['Tipo'] == 3:        
                count[2] = count[2]+1
                
        def max_num_in_list_label(list):
            maxpos = list.index(max(list)) +1
            return labels[maxpos]
        
        #getting the label and verifying with the class label in xq
        if max_num_in_list_label(count) in xq.values:
            xqValues = xq.values
            xqValues2 = max_num_in_list_label(count)
            print("Prueba Positiva")
            print(xqValues[0][0])
            matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] = matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] + 1
        else:
            xqValues = xq.values
            print(xqValues)
            xqValues2 = max_num_in_list_label(count)
            print(xqValues2)
            print("Prueva Negativa")
            print(xqValues[0][0])
            matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] = matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] + 1
        print("----------------------------------------------------------------------------")

    Presicion = []
    Sensibilidad = []

    sumaFilas =  [sum(i) for i in matriz]
    sumaColumnas =  [sum(i) for i in zip(*matriz)]
    print("Matriz")
    print(matriz[0])
    print(matriz[1])
    print(matriz[2])
    print("Sumatoria de Columnas")      
    print(sumaColumnas)
    print("Sumatoria de Filas")
    print(sumaFilas)

    print("Presicion")
    for i in range(n):
        for j in range(m):
            if(j == i):
                if(matriz[i][j] == 0 and  sumaFilas[i] == 0):
                    Presicion.append("Infinito")
                else:
                    Presicion.append((matriz[i][j]/sumaFilas[i])*100)
    print(Presicion)
    print("Sensibilidad")
    for i in range(n):
        for j in range(m):
            if(j == i):
                if(matriz[i][j] == 0 and  sumaFilas[i] == 0):
                    Sensibilidad.append("Infinito")
                else:
                    Sensibilidad.append((matriz[i][j]/sumaColumnas[i])*100)
    print(Sensibilidad) 
  
def Tabla8(Entrada, Entrada2,Entrada3,Entrada4,Entrada5,Entrada6,Entrada7,Entrada8, df, repeticiones): 
    n = 3
    m = 3

    matriz = []

    for i in range(n):
        matriz.append([])
        for j in range(m):
            matriz[i].append(0)

    print(matriz)
    for i in range (repeticiones):
        print("----------------------------------------------------------------------------")
        print(type(df))
        print(df.shape)
        pruebita2 = df.describe()
        labels = dict(zip(df.Tipo.unique(), df.Tipo.unique()))
        print(df['Tipo'].value_counts())

        #taking random record and storing in xq
        xq = df.sample()

        # droping the xq from data using index value
        df.drop(xq.index, inplace=True)
        print(df.shape)
        xq_final = pd.DataFrame(xq[[Entrada, Entrada2, Entrada3, Entrada4, Entrada5, Entrada6, Entrada7, Entrada8]])
        final = xq_final

        # calculating ecludian distance
        def cal_distance(x):      
            a = x.to_numpy()
            b = xq_final.to_numpy()    
            distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(a, b[0])]))
            return distance

        # calculating distance
        df['distance'] = df[[Entrada, Entrada2, Entrada3, Entrada4, Entrada5, Entrada6, Entrada7, Entrada8]].apply(cal_distance, axis=1)

        #sorting the values based on distance
        df_sort = df.sort_values('distance',ascending=True)

        # taking top 11 records because k is 11
        df_after_sort = df_sort.head(11)

        pruebita3 = df_after_sort.reset_index()

        print(df_after_sort.iloc[0])

        #q esta pasando aki?
        count = [0 for i in range(0, len(df['Tipo'].unique()))]
        for xi in range(0, len(df_after_sort)):       
            if df_after_sort.iloc[xi]['Tipo'] == 1:        
                count[0] = count[0]+1
            elif df_after_sort.iloc[xi]['Tipo'] == 2:        
                count[1] = count[1]+1
            elif df_after_sort.iloc[xi]['Tipo'] == 3:        
                count[2] = count[2]+1
                
        def max_num_in_list_label(list):
            maxpos = list.index(max(list)) +1
            return labels[maxpos]
        
        #getting the label and verifying with the class label in xq
        if max_num_in_list_label(count) in xq.values:
            xqValues = xq.values
            xqValues2 = max_num_in_list_label(count)
            print("Prueba Positiva")
            print(xqValues[0][0])
            matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] = matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] + 1
        else:
            xqValues = xq.values
            print(xqValues)
            xqValues2 = max_num_in_list_label(count)
            print(xqValues2)
            print("Prueva Negativa")
            print(xqValues[0][0])
            matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] = matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] + 1
        print("----------------------------------------------------------------------------")

    Presicion = []
    Sensibilidad = []

    sumaFilas =  [sum(i) for i in matriz]
    sumaColumnas =  [sum(i) for i in zip(*matriz)]
    print("Matriz")
    print(matriz[0])
    print(matriz[1])
    print(matriz[2])
    print("Sumatoria de Columnas")      
    print(sumaColumnas)
    print("Sumatoria de Filas")
    print(sumaFilas)

    print("Presicion")
    for i in range(n):
        for j in range(m):
            if(j == i):
                if(matriz[i][j] == 0 and  sumaFilas[i] == 0):
                    Presicion.append("Infinito")
                else:
                    Presicion.append((matriz[i][j]/sumaFilas[i])*100)
    print(Presicion)
    print("Sensibilidad")
    for i in range(n):
        for j in range(m):
            if(j == i):
                if(matriz[i][j] == 0 and  sumaFilas[i] == 0):
                    Sensibilidad.append("Infinito")
                else:
                    Sensibilidad.append((matriz[i][j]/sumaColumnas[i])*100)
    print(Sensibilidad)
    
def Tabla9(Entrada, Entrada2,Entrada3,Entrada4,Entrada5,Entrada6,Entrada7,Entrada8, Entrada9, df, repeticiones): 
    n = 3
    m = 3

    matriz = []

    for i in range(n):
        matriz.append([])
        for j in range(m):
            matriz[i].append(0)

    print(matriz)
    for i in range (repeticiones):
        print("----------------------------------------------------------------------------")
        print(type(df))
        print(df.shape)
        pruebita2 = df.describe()
        labels = dict(zip(df.Tipo.unique(), df.Tipo.unique()))
        print(df['Tipo'].value_counts())

        #taking random record and storing in xq
        xq = df.sample()

        # droping the xq from data using index value
        df.drop(xq.index, inplace=True)
        print(df.shape)
        xq_final = pd.DataFrame(xq[[Entrada, Entrada2, Entrada3, Entrada4, Entrada5, Entrada6, Entrada7, Entrada8, Entrada9]])
        final = xq_final

        # calculating ecludian distance
        def cal_distance(x):      
            a = x.to_numpy()
            b = xq_final.to_numpy()    
            distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(a, b[0])]))
            return distance

        # calculating distance
        df['distance'] = df[[Entrada, Entrada2, Entrada3, Entrada4, Entrada5, Entrada6, Entrada7, Entrada8, Entrada9]].apply(cal_distance, axis=1)

        #sorting the values based on distance
        df_sort = df.sort_values('distance',ascending=True)

        # taking top 11 records because k is 11
        df_after_sort = df_sort.head(11)

        pruebita3 = df_after_sort.reset_index()

        print(df_after_sort.iloc[0])

        #q esta pasando aki?
        count = [0 for i in range(0, len(df['Tipo'].unique()))]
        for xi in range(0, len(df_after_sort)):       
            if df_after_sort.iloc[xi]['Tipo'] == 1:        
                count[0] = count[0]+1
            elif df_after_sort.iloc[xi]['Tipo'] == 2:        
                count[1] = count[1]+1
            elif df_after_sort.iloc[xi]['Tipo'] == 3:        
                count[2] = count[2]+1
                
        def max_num_in_list_label(list):
            maxpos = list.index(max(list)) +1
            return labels[maxpos]
        
        #getting the label and verifying with the class label in xq
        if max_num_in_list_label(count) in xq.values:
            xqValues = xq.values
            xqValues2 = max_num_in_list_label(count)
            print("Prueba Positiva")
            print(xqValues[0][0])
            matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] = matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] + 1
        else:
            xqValues = xq.values
            print(xqValues)
            xqValues2 = max_num_in_list_label(count)
            print(xqValues2)
            print("Prueva Negativa")
            print(xqValues[0][0])
            matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] = matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] + 1
        print("----------------------------------------------------------------------------")

    Presicion = []
    Sensibilidad = []

    sumaFilas =  [sum(i) for i in matriz]
    sumaColumnas =  [sum(i) for i in zip(*matriz)]
    print("Matriz")
    print(matriz[0])
    print(matriz[1])
    print(matriz[2])
    print("Sumatoria de Columnas")      
    print(sumaColumnas)
    print("Sumatoria de Filas")
    print(sumaFilas)

    print("Presicion")
    for i in range(n):
        for j in range(m):
            if(j == i):
                if(matriz[i][j] == 0 and  sumaFilas[i] == 0):
                    Presicion.append("Infinito")
                else:
                    Presicion.append((matriz[i][j]/sumaFilas[i])*100)
    print(Presicion)
    print("Sensibilidad")
    for i in range(n):
        for j in range(m):
            if(j == i):
                if(matriz[i][j] == 0 and  sumaFilas[i] == 0):
                    Sensibilidad.append("Infinito")
                else:
                    Sensibilidad.append((matriz[i][j]/sumaColumnas[i])*100)
    print(Sensibilidad)
    
def Tabla10(Entrada, Entrada2,Entrada3,Entrada4,Entrada5,Entrada6,Entrada7,Entrada8, Entrada9,Entrada10, df, repeticiones): 
    n = 3
    m = 3

    matriz = []

    for i in range(n):
        matriz.append([])
        for j in range(m):
            matriz[i].append(0)

    print(matriz)
    for i in range (repeticiones):
        print("----------------------------------------------------------------------------")
        print(type(df))
        print(df.shape)
        pruebita2 = df.describe()
        labels = dict(zip(df.Tipo.unique(), df.Tipo.unique()))
        print(df['Tipo'].value_counts())

        #taking random record and storing in xq
        xq = df.sample()

        # droping the xq from data using index value
        df.drop(xq.index, inplace=True)
        print(df.shape)
        xq_final = pd.DataFrame(xq[[Entrada, Entrada2, Entrada3, Entrada4, Entrada5, Entrada6, Entrada7, Entrada8, Entrada9, Entrada10]])
        final = xq_final

        # calculating ecludian distance
        def cal_distance(x):      
            a = x.to_numpy()
            b = xq_final.to_numpy()    
            distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(a, b[0])]))
            return distance

        # calculating distance
        df['distance'] = df[[Entrada, Entrada2, Entrada3, Entrada4, Entrada5, Entrada6, Entrada7, Entrada8, Entrada9, Entrada10]].apply(cal_distance, axis=1)

        #sorting the values based on distance
        df_sort = df.sort_values('distance',ascending=True)

        # taking top 11 records because k is 11
        df_after_sort = df_sort.head(11)

        pruebita3 = df_after_sort.reset_index()

        print(df_after_sort.iloc[0])

        #q esta pasando aki?
        count = [0 for i in range(0, len(df['Tipo'].unique()))]
        for xi in range(0, len(df_after_sort)):       
            if df_after_sort.iloc[xi]['Tipo'] == 1:        
                count[0] = count[0]+1
            elif df_after_sort.iloc[xi]['Tipo'] == 2:        
                count[1] = count[1]+1
            elif df_after_sort.iloc[xi]['Tipo'] == 3:        
                count[2] = count[2]+1
                
        def max_num_in_list_label(list):
            maxpos = list.index(max(list)) +1
            return labels[maxpos]
        
        #getting the label and verifying with the class label in xq
        if max_num_in_list_label(count) in xq.values:
            xqValues = xq.values
            xqValues2 = max_num_in_list_label(count)
            print("Prueba Positiva")
            print(xqValues[0][0])
            matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] = matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] + 1
        else:
            xqValues = xq.values
            print(xqValues)
            xqValues2 = max_num_in_list_label(count)
            print(xqValues2)
            print("Prueva Negativa")
            print(xqValues[0][0])
            matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] = matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] + 1
        print("----------------------------------------------------------------------------")

    Presicion = []
    Sensibilidad = []

    sumaFilas =  [sum(i) for i in matriz]
    sumaColumnas =  [sum(i) for i in zip(*matriz)]
    print("Matriz")
    print(matriz[0])
    print(matriz[1])
    print(matriz[2])
    print("Sumatoria de Columnas")      
    print(sumaColumnas)
    print("Sumatoria de Filas")
    print(sumaFilas)

    print("Presicion")
    for i in range(n):
        for j in range(m):
            if(j == i):
                if(matriz[i][j] == 0 and  sumaFilas[i] == 0):
                    Presicion.append("Infinito")
                else:
                    Presicion.append((matriz[i][j]/sumaFilas[i])*100)
    print(Presicion)
    print("Sensibilidad")
    for i in range(n):
        for j in range(m):
            if(j == i):
                if(matriz[i][j] == 0 and  sumaFilas[i] == 0):
                    Sensibilidad.append("Infinito")
                else:
                    Sensibilidad.append((matriz[i][j]/sumaColumnas[i])*100)
    print(Sensibilidad)  
    
def Tabla11(Entrada, Entrada2,Entrada3,Entrada4,Entrada5,Entrada6,Entrada7,Entrada8, Entrada9,Entrada10, Entrada11, df, repeticiones): 
    n = 3
    m = 3

    matriz = []

    for i in range(n):
        matriz.append([])
        for j in range(m):
            matriz[i].append(0)

    print(matriz)
    for i in range (repeticiones):
        print("----------------------------------------------------------------------------")
        print(type(df))
        print(df.shape)
        pruebita2 = df.describe()
        labels = dict(zip(df.Tipo.unique(), df.Tipo.unique()))
        print(df['Tipo'].value_counts())

        #taking random record and storing in xq
        xq = df.sample()

        # droping the xq from data using index value
        df.drop(xq.index, inplace=True)
        print(df.shape)
        xq_final = pd.DataFrame(xq[[Entrada, Entrada2, Entrada3, Entrada4, Entrada5, Entrada6, Entrada7, Entrada8, Entrada9, Entrada10, Entrada11]])
        final = xq_final

        # calculating ecludian distance
        def cal_distance(x):      
            a = x.to_numpy()
            b = xq_final.to_numpy()    
            distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(a, b[0])]))
            return distance

        # calculating distance
        df['distance'] = df[[Entrada, Entrada2, Entrada3, Entrada4, Entrada5, Entrada6, Entrada7, Entrada8, Entrada9, Entrada10, Entrada11]].apply(cal_distance, axis=1)

        #sorting the values based on distance
        df_sort = df.sort_values('distance',ascending=True)

        # taking top 11 records because k is 11
        df_after_sort = df_sort.head(11)

        pruebita3 = df_after_sort.reset_index()

        print(df_after_sort.iloc[0])

        #q esta pasando aki?
        count = [0 for i in range(0, len(df['Tipo'].unique()))]
        for xi in range(0, len(df_after_sort)):       
            if df_after_sort.iloc[xi]['Tipo'] == 1:        
                count[0] = count[0]+1
            elif df_after_sort.iloc[xi]['Tipo'] == 2:        
                count[1] = count[1]+1
            elif df_after_sort.iloc[xi]['Tipo'] == 3:        
                count[2] = count[2]+1
                
        def max_num_in_list_label(list):
            maxpos = list.index(max(list)) +1
            return labels[maxpos]
        
        #getting the label and verifying with the class label in xq
        if max_num_in_list_label(count) in xq.values:
            xqValues = xq.values
            xqValues2 = max_num_in_list_label(count)
            print("Prueba Positiva")
            print(xqValues[0][0])
            matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] = matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] + 1
        else:
            xqValues = xq.values
            print(xqValues)
            xqValues2 = max_num_in_list_label(count)
            print(xqValues2)
            print("Prueva Negativa")
            print(xqValues[0][0])
            matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] = matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] + 1
        print("----------------------------------------------------------------------------")

    Presicion = []
    Sensibilidad = []

    sumaFilas =  [sum(i) for i in matriz]
    sumaColumnas =  [sum(i) for i in zip(*matriz)]
    print("Matriz")
    print(matriz[0])
    print(matriz[1])
    print(matriz[2])
    print("Sumatoria de Columnas")      
    print(sumaColumnas)
    print("Sumatoria de Filas")
    print(sumaFilas)

    print("Presicion")
    for i in range(n):
        for j in range(m):
            if(j == i):
                if(matriz[i][j] == 0 and  sumaFilas[i] == 0):
                    Presicion.append("Infinito")
                else:
                    Presicion.append((matriz[i][j]/sumaFilas[i])*100)
    print(Presicion)
    print("Sensibilidad")
    for i in range(n):
        for j in range(m):
            if(j == i):
                if(matriz[i][j] == 0 and  sumaFilas[i] == 0):
                    Sensibilidad.append("Infinito")
                else:
                    Sensibilidad.append((matriz[i][j]/sumaColumnas[i])*100)
    print(Sensibilidad)

def Tabla12(Entrada, Entrada2,Entrada3,Entrada4,Entrada5,Entrada6,Entrada7,Entrada8, Entrada9,Entrada10, Entrada11, Entrada12, df, repeticiones): 
    n = 3
    m = 3

    matriz = []

    for i in range(n):
        matriz.append([])
        for j in range(m):
            matriz[i].append(0)

    print(matriz)
    for i in range (repeticiones):
        print("----------------------------------------------------------------------------")
        print(type(df))
        print(df.shape)
        pruebita2 = df.describe()
        labels = dict(zip(df.Tipo.unique(), df.Tipo.unique()))
        print(df['Tipo'].value_counts())

        #taking random record and storing in xq
        xq = df.sample()

        # droping the xq from data using index value
        df.drop(xq.index, inplace=True)
        print(df.shape)
        xq_final = pd.DataFrame(xq[[Entrada, Entrada2, Entrada3, Entrada4, Entrada5, Entrada6, Entrada7, Entrada8, Entrada9, Entrada10, Entrada11, Entrada12]])
        final = xq_final

        # calculating ecludian distance
        def cal_distance(x):      
            a = x.to_numpy()
            b = xq_final.to_numpy()    
            distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(a, b[0])]))
            return distance

        # calculating distance
        df['distance'] = df[[Entrada, Entrada2, Entrada3, Entrada4, Entrada5, Entrada6, Entrada7, Entrada8, Entrada9, Entrada10, Entrada11, Entrada12]].apply(cal_distance, axis=1)

        #sorting the values based on distance
        df_sort = df.sort_values('distance',ascending=True)

        # taking top 11 records because k is 11
        df_after_sort = df_sort.head(11)

        pruebita3 = df_after_sort.reset_index()

        print(df_after_sort.iloc[0])

        #q esta pasando aki?
        count = [0 for i in range(0, len(df['Tipo'].unique()))]
        for xi in range(0, len(df_after_sort)):       
            if df_after_sort.iloc[xi]['Tipo'] == 1:        
                count[0] = count[0]+1
            elif df_after_sort.iloc[xi]['Tipo'] == 2:        
                count[1] = count[1]+1
            elif df_after_sort.iloc[xi]['Tipo'] == 3:        
                count[2] = count[2]+1
                
        def max_num_in_list_label(list):
            maxpos = list.index(max(list)) +1
            return labels[maxpos]
        
        #getting the label and verifying with the class label in xq
        if max_num_in_list_label(count) in xq.values:
            xqValues = xq.values
            xqValues2 = max_num_in_list_label(count)
            print("Prueba Positiva")
            print(xqValues[0][0])
            matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] = matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] + 1
        else:
            xqValues = xq.values
            print(xqValues)
            xqValues2 = max_num_in_list_label(count)
            print(xqValues2)
            print("Prueva Negativa")
            print(xqValues[0][0])
            matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] = matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] + 1
        print("----------------------------------------------------------------------------")

    Presicion = []
    Sensibilidad = []

    sumaFilas =  [sum(i) for i in matriz]
    sumaColumnas =  [sum(i) for i in zip(*matriz)]
    print("Matriz")
    print(matriz[0])
    print(matriz[1])
    print(matriz[2])
    print("Sumatoria de Columnas")      
    print(sumaColumnas)
    print("Sumatoria de Filas")
    print(sumaFilas)

    print("Presicion")
    for i in range(n):
        for j in range(m):
            if(j == i):
                if(matriz[i][j] == 0 and  sumaFilas[i] == 0):
                    Presicion.append("Infinito")
                else:
                    Presicion.append((matriz[i][j]/sumaFilas[i])*100)
    print(Presicion)
    print("Sensibilidad")
    for i in range(n):
        for j in range(m):
            if(j == i):
                if(matriz[i][j] == 0 and  sumaFilas[i] == 0):
                    Sensibilidad.append("Infinito")
                else:
                    Sensibilidad.append((matriz[i][j]/sumaColumnas[i])*100)
    print(Sensibilidad)
    
def Tabla13(Entrada, Entrada2,Entrada3,Entrada4,Entrada5,Entrada6,Entrada7,Entrada8, Entrada9,Entrada10, Entrada11, Entrada12, Entrada13, df, repeticiones): 
    n = 3
    m = 3

    matriz = []

    for i in range(n):
        matriz.append([])
        for j in range(m):
            matriz[i].append(0)

    print(matriz)
    for i in range (repeticiones):
        print("----------------------------------------------------------------------------")
        print(type(df))
        print(df.shape)
        pruebita2 = df.describe()
        labels = dict(zip(df.Tipo.unique(), df.Tipo.unique()))
        print(df['Tipo'].value_counts())

        #taking random record and storing in xq
        xq = df.sample()

        # droping the xq from data using index value
        df.drop(xq.index, inplace=True)
        print(df.shape)
        xq_final = pd.DataFrame(xq[[Entrada, Entrada2, Entrada3, Entrada4, Entrada5, Entrada6, Entrada7, Entrada8, Entrada9, Entrada10, Entrada11, Entrada12, Entrada13]])
        final = xq_final

        # calculating ecludian distance
        def cal_distance(x):      
            a = x.to_numpy()
            b = xq_final.to_numpy()    
            distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(a, b[0])]))
            return distance

        # calculating distance
        df['distance'] = df[[Entrada, Entrada2, Entrada3, Entrada4, Entrada5, Entrada6, Entrada7, Entrada8, Entrada9, Entrada10, Entrada11, Entrada12, Entrada13]].apply(cal_distance, axis=1)

        #sorting the values based on distance
        df_sort = df.sort_values('distance',ascending=True)

        # taking top 11 records because k is 11
        df_after_sort = df_sort.head(11)

        pruebita3 = df_after_sort.reset_index()

        print(df_after_sort.iloc[0])

        #q esta pasando aki?
        count = [0 for i in range(0, len(df['Tipo'].unique()))]
        for xi in range(0, len(df_after_sort)):       
            if df_after_sort.iloc[xi]['Tipo'] == 1:        
                count[0] = count[0]+1
            elif df_after_sort.iloc[xi]['Tipo'] == 2:        
                count[1] = count[1]+1
            elif df_after_sort.iloc[xi]['Tipo'] == 3:        
                count[2] = count[2]+1
                
        def max_num_in_list_label(list):
            maxpos = list.index(max(list)) +1
            return labels[maxpos]
        
        #getting the label and verifying with the class label in xq
        if max_num_in_list_label(count) in xq.values:
            xqValues = xq.values
            xqValues2 = max_num_in_list_label(count)
            print("Prueba Positiva")
            print(xqValues[0][0])
            matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] = matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] + 1
        else:
            xqValues = xq.values
            print(xqValues)
            xqValues2 = max_num_in_list_label(count)
            print(xqValues2)
            print("Prueva Negativa")
            print(xqValues[0][0])
            matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] = matriz[int(xqValues[0][0] - 1)][xqValues2 - 1] + 1
        print("----------------------------------------------------------------------------")

    Presicion = []
    Sensibilidad = []

    sumaFilas =  [sum(i) for i in matriz]
    sumaColumnas =  [sum(i) for i in zip(*matriz)]
    print("Matriz")
    print(matriz[0])
    print(matriz[1])
    print(matriz[2])
    print("Sumatoria de Columnas")      
    print(sumaColumnas)
    print("Sumatoria de Filas")
    print(sumaFilas)

    print("Presicion")
    for i in range(n):
        for j in range(m):
            if(j == i):
                if(matriz[i][j] == 0 and  sumaFilas[i] == 0):
                    Presicion.append("Infinito")
                else:
                    Presicion.append((matriz[i][j]/sumaFilas[i])*100)
    print(Presicion)
    print("Sensibilidad")
    for i in range(n):
        for j in range(m):
            if(j == i):
                if(matriz[i][j] == 0 and  sumaFilas[i] == 0):
                    Sensibilidad.append("Infinito")
                else:
                    Sensibilidad.append((matriz[i][j]/sumaColumnas[i])*100)
    print(Sensibilidad)
    
def Regrasion():
    k = ComboCantidad.get() 
    Entrada = ComboRE1.get()
    Entrada2 = ComboRE2.get()
    Entrada3 = ComboRE3.get()
    Entrada4 = ComboRE4.get()
    Entrada5 = ComboRE5.get()
    Entrada6 = ComboRE6.get()
    Entrada7 = ComboRE7.get()
    Entrada8 = ComboRE8.get()
    Entrada9 = ComboRE9.get()
    Entrada10 = ComboRE10.get()
    Entrada11 = ComboRE11.get()
    Entrada12 = ComboRE12.get()
    Entrada13 = ComboRE13.get()
    repe = ComboRepeticion.get()
    repeticiones = int(repe)
    
    if(k == "2"):
        Tabla2(Entrada, Entrada2, datos, repeticiones)
    elif(k == "3"):
        Tabla3(Entrada, Entrada2, Entrada3, datos, repeticiones)
    elif(k == "4"):
        Tabla4(Entrada, Entrada2, Entrada3, Entrada4, datos, repeticiones)
    elif(k == "5"):
        Tabla5(Entrada, Entrada2, Entrada3, Entrada4, Entrada5, datos, repeticiones)
    elif(k == "6"):
        Tabla6(Entrada, Entrada2, Entrada3, Entrada4, Entrada5, Entrada6, datos, repeticiones)
    elif(k == "7"):
        Tabla7(Entrada, Entrada2, Entrada3, Entrada4, Entrada5, Entrada6, Entrada7, datos, repeticiones)
    elif(k == "8"):
        Tabla8(Entrada, Entrada2, Entrada3, Entrada4, Entrada5, Entrada6, Entrada7, Entrada8, datos, repeticiones)
    elif(k == "9"):
        Tabla9(Entrada, Entrada2, Entrada3, Entrada4, Entrada5, Entrada6, Entrada7, Entrada8, Entrada9, datos, repeticiones)
    elif(k == "10"):
        Tabla10(Entrada, Entrada2, Entrada3, Entrada4, Entrada5, Entrada6, Entrada7, Entrada8, Entrada9, Entrada10, datos, repeticiones)
    elif(k == "11"):
        Tabla11(Entrada, Entrada2, Entrada3, Entrada4, Entrada5, Entrada6, Entrada7, Entrada8, Entrada9, Entrada10, Entrada11, datos, repeticiones)
    elif(k == "12"):
        Tabla12(Entrada, Entrada2, Entrada3, Entrada4, Entrada5, Entrada6, Entrada7, Entrada8, Entrada9, Entrada10, Entrada11, Entrada12, datos, repeticiones)
    elif(k == "13"):    
        Tabla13(Entrada, Entrada2, Entrada3, Entrada4, Entrada5, Entrada6, Entrada7, Entrada8, Entrada9, Entrada10, Entrada11, Entrada12, Entrada13, datos, repeticiones) 
        
def show_selection():
    Opciones = ['alcohol', 'acido malico','Ceniza','alcalinidad de la ceniza','magnesio', "fenoles totales", "flavonoides", "fenoles no flavonoides", "proantocianinas", "fntensidad de color", "tonalidad", "vinos diluidos", "prolina"]
    Opciones2 = ["1", "2", "3"]
    k = ComboCantidad.get()
    ComboRE1["values"] = []
    ComboRE2["values"] = []
    ComboRE3["values"] = []
    ComboRE4["values"] = []
    ComboRE5["values"] = []
    ComboRE6["values"] = []
    ComboRE7["values"] = []
    ComboRE8["values"] = []
    ComboRE9["values"] = []
    ComboRE10["values"] = []
    ComboRE11["values"] = []
    ComboRE12["values"] = []
    ComboRE1.set("")
    ComboRE2.set("")
    ComboRE3.set("")
    ComboRE4.set("")
    ComboRE5.set("")
    ComboRE6.set("")
    ComboRE7.set("")
    ComboRE8.set("")
    ComboRE9.set("")
    ComboRE10.set("")
    ComboRE11.set("")
    ComboRE12.set("")
    
    if(k == "1"):
        ComboRE1["values"] = Opciones
    elif(k == "2"):
        ComboRE1["values"] = Opciones
        ComboRE2["values"] = Opciones
    elif(k == "3"):
        ComboRE1["values"] = Opciones
        ComboRE2["values"] = Opciones
        ComboRE3["values"] = Opciones
    elif(k == "4"):
        ComboRE1["values"] = Opciones
        ComboRE2["values"] = Opciones
        ComboRE3["values"] = Opciones
        ComboRE4["values"] = Opciones
    elif(k == "5"):
        ComboRE1["values"] = Opciones
        ComboRE2["values"] = Opciones
        ComboRE3["values"] = Opciones
        ComboRE4["values"] = Opciones
        ComboRE5["values"] = Opciones
    elif(k == "6"):
        ComboRE1["values"] = Opciones
        ComboRE2["values"] = Opciones
        ComboRE3["values"] = Opciones
        ComboRE4["values"] = Opciones
        ComboRE5["values"] = Opciones
        ComboRE6["values"] = Opciones
    elif(k == "7"):
        ComboRE1["values"] = Opciones
        ComboRE2["values"] = Opciones
        ComboRE3["values"] = Opciones
        ComboRE4["values"] = Opciones
        ComboRE5["values"] = Opciones
        ComboRE6["values"] = Opciones
        ComboRE7["values"] = Opciones
    elif(k == "8"):
        ComboRE1["values"] = Opciones
        ComboRE2["values"] = Opciones
        ComboRE3["values"] = Opciones
        ComboRE4["values"] = Opciones
        ComboRE5["values"] = Opciones
        ComboRE6["values"] = Opciones
        ComboRE7["values"] = Opciones
        ComboRE8["values"] = Opciones
    elif(k == "9"):
        ComboRE1["values"] = Opciones
        ComboRE2["values"] = Opciones
        ComboRE3["values"] = Opciones
        ComboRE4["values"] = Opciones
        ComboRE5["values"] = Opciones
        ComboRE6["values"] = Opciones
        ComboRE7["values"] = Opciones
        ComboRE8["values"] = Opciones
        ComboRE9["values"] = Opciones
    elif(k == "10"):
        ComboRE1["values"] = Opciones
        ComboRE2["values"] = Opciones
        ComboRE3["values"] = Opciones
        ComboRE4["values"] = Opciones
        ComboRE5["values"] = Opciones
        ComboRE6["values"] = Opciones
        ComboRE7["values"] = Opciones
        ComboRE8["values"] = Opciones
        ComboRE9["values"] = Opciones
        ComboRE10["values"] = Opciones
    elif(k == "11"):
        ComboRE1["values"] = Opciones
        ComboRE2["values"] = Opciones
        ComboRE3["values"] = Opciones
        ComboRE4["values"] = Opciones
        ComboRE5["values"] = Opciones
        ComboRE6["values"] = Opciones
        ComboRE7["values"] = Opciones
        ComboRE8["values"] = Opciones
        ComboRE9["values"] = Opciones
        ComboRE10["values"] = Opciones
        ComboRE11["values"] = Opciones
    elif(k == "12"):
        ComboRE1["values"] = Opciones
        ComboRE2["values"] = Opciones
        ComboRE3["values"] = Opciones
        ComboRE4["values"] = Opciones
        ComboRE5["values"] = Opciones
        ComboRE6["values"] = Opciones
        ComboRE7["values"] = Opciones
        ComboRE8["values"] = Opciones
        ComboRE9["values"] = Opciones
        ComboRE10["values"] = Opciones
        ComboRE11["values"] = Opciones
        ComboRE12["values"] = Opciones
    elif(k == "13"):
        ComboRE1["values"] = Opciones
        ComboRE2["values"] = Opciones
        ComboRE3["values"] = Opciones
        ComboRE4["values"] = Opciones
        ComboRE5["values"] = Opciones
        ComboRE6["values"] = Opciones
        ComboRE7["values"] = Opciones
        ComboRE8["values"] = Opciones
        ComboRE9["values"] = Opciones
        ComboRE10["values"] = Opciones
        ComboRE11["values"] = Opciones
        ComboRE12["values"] = Opciones
        ComboRE13["values"] = Opciones


#----------------------------------------- Vista --------------------------#
main_window.config(width=1200, height=700)
main_window.title("Parcial N°2")
main_window.resizable(width = False, height = False)
fondo = tk.PhotoImage(file= "Fondo.PNG")
fondo1 = tk.Label(main_window, image=fondo).place(x=0, y=0, width=1200, height=700)

ComboCantidad = ttk.Combobox(
    state="readonly",
    values=["1","2","3","4","5","6","7","8","9","10","11","12","13"]
)

ComboCantidad.place(x=600, y=160)
ComboCantidad.configure(width=20, height=10)

ComboRepeticion = ttk.Combobox(
    state="readonly",
    values=["10","20","30","40","50","60","70","80","90","100"]
)

ComboRepeticion.place(x=600, y=100)
ComboRepeticion.configure(width=20, height=10)

ComboRE1 = ttk.Combobox(
    state="readonly",
    values=[]
)

ComboRE1.place(x=150, y=100)
ComboRE1.configure(width=20, height=10)

ComboRE2 = ttk.Combobox(
    state="readonly",
    values=[]
)

ComboRE2.place(x=150, y=128)
ComboRE2.configure(width=20, height=10)

ComboRE3 = ttk.Combobox(
    state="readonly",
    values=[]
)

ComboRE3.place(x=150, y=156)
ComboRE3.configure(width=20, height=10)

ComboRE4 = ttk.Combobox(
    state="readonly",
    values=[]
)

ComboRE4.place(x=150, y=184)
ComboRE4.configure(width=20, height=10)

ComboRE5 = ttk.Combobox(
    state="readonly",
    values=[]
)

ComboRE5.place(x=150, y=212)
ComboRE5.configure(width=20, height=10)

ComboRE6 = ttk.Combobox(
    state="readonly",
    values=[]
)

ComboRE6.place(x=150, y=240)
ComboRE6.configure(width=20, height=10)

ComboRE7 = ttk.Combobox(
    state="readonly",
    values=[]
)

ComboRE7.place(x=150, y=268)
ComboRE7.configure(width=20, height=10)

ComboRE8 = ttk.Combobox(
    state="readonly",
    values=[]
)

ComboRE8.place(x=150, y=296)
ComboRE8.configure(width=20, height=10)

ComboRE9 = ttk.Combobox(
    state="readonly",
    values=[]
)

ComboRE9.place(x=150, y=324)
ComboRE9.configure(width=20, height=10)

ComboRE10 = ttk.Combobox(
    state="readonly",
    values=[]
)

ComboRE10.place(x=150, y=352)
ComboRE10.configure(width=20, height=10)

ComboRE11 = ttk.Combobox(
    state="readonly",
    values=[]
)

ComboRE11.place(x=150, y=380)
ComboRE11.configure(width=20, height=10)

ComboRE12 = ttk.Combobox(
    state="readonly",
    values=[]
)

ComboRE12.place(x=150, y=408)
ComboRE12.configure(width=20, height=10)

ComboRE13 = ttk.Combobox(
    state="readonly",
    values=[]
)

ComboRE13.place(x=150, y=436)
ComboRE13.configure(width=20, height=10)


#--------------------------------- Controlador -------------------------------#

button = tk.Button(text="Actualizar Entrada", command=show_selection, width= 34, height=3)
button.place(x=550, y=200)

button = tk.Button(text="Hallar Regresion", command=Regrasion, width= 28, height=2)
button.place(x=120, y=480)

main_window.mainloop()      
