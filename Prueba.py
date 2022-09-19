import numpy as np 
import pandas as pd 
import math

url= "wine - copia.csv"
df = pd.read_csv(url)
columnas = ['Tipo', 'Alcohol', 'Acido malico','Ceniza','Alcalinidad de la ceniza','Magnesio', "Fenoles totales", "Flavonoides", "Fenoles no flavonoides", "Proantocianinas", "Intensidad de color", "Tonalidad", "Vinos diluidos", "Prolina"]
df.columns = columnas
pruebita1 = df.head()

n = 3
m = 3

matriz = []

for i in range(n):
    matriz.append([])
    for j in range(m):
        matriz[i].append(0)

print(matriz)
for i in range (0,10):
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
    xq_final = pd.DataFrame(xq[['Alcohol', 'Acido malico','Ceniza','Alcalinidad de la ceniza','Magnesio', "Fenoles totales", "Flavonoides", "Fenoles no flavonoides", "Proantocianinas", "Intensidad de color", "Tonalidad", "Vinos diluidos", "Prolina"]])
    final = xq_final

    # calculating ecludian distance
    def cal_distance(x):      
        a = x.to_numpy()
        b = xq_final.to_numpy()    
        distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(a, b[0])]))
        return distance

    # calculating distance
    df['distance'] = df[['Alcohol', 'Acido malico','Ceniza','Alcalinidad de la ceniza','Magnesio', "Fenoles totales", "Flavonoides", "Fenoles no flavonoides", "Proantocianinas", "Intensidad de color", "Tonalidad", "Vinos diluidos", "Prolina"]].apply(cal_distance, axis=1)

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
