# -*- coding: utf-8 -*-
"""
@author: AnaMaria
"""

#----------- Librerias Utlizadas ------------------------

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import operator


#---------- Funciones ---------------------------------

#Funciones para encontrar punto de inflexion,
#Rotacion de plano, con esto se facilita encontrar la distancia 
#entre la grafica y la linea recta (x[0],y[0])(x[-1],y[-1])
def get_data_radiant(data):
  return np.arctan2(data[:, 1].max() - data[:, 1].min(), 
                    data[:, 0].max() - data[:, 0].min())
def find_elbow(data, theta):

    # make rotation matrix
    co = np.cos(-theta)
    si = np.sin(-theta)
    rotation_matrix = np.array(((co, -si), (si, co)))

    # rotate data vector
    rotated_vector = data.dot(rotation_matrix)
    
    # return index of elbow
    return np.where(rotated_vector[:,1] == rotated_vector[:,1].min())[0][0]


#---------- Definicion de variables ---------------------

#DataFrames
centroids=pd.read_excel("excel/centroids2.xlsx",index_col=0) # DF con los centroides de la agrupacion elegida
bd=pd.read_excel("BaseDatos_Normalizada.xlsx",index_col=0) # DF con la BD normalizada
bd_new= pd.DataFrame() # DF donde se almacenaran las columnas significativas
bd_new['a']=range(0,len(bd)) # Se especifica el numero de registros con que se trabajara

#Listas
variables=bd.columns.values.ravel().tolist() #Lista con los nombres de las variables
var_orden=[] #Lista de variables ordenandas de mayor a menor diferencia entre centroides

#Variables
cen_min=centroids.min(axis=0) #Valor minimo del centroide por variable
cen_max=centroids.max(axis=0) #Valor maximoo del centroide por variable


#---------- Diferencia maxima entre centroides por variable ---------------

diferencia=(cen_max-cen_min).tolist() # Diferencia entre centroides
dif=sorted(diferencia,reverse=True) # Diferencia entre centroides de mayor a menor
orden=dict(zip(variables,diferencia)) # Diccionario con variables y su maxima diferencia de centroides
var_orden=sorted(orden.items(),key=operator.itemgetter(1),reverse=True) #Lista de variables ordenandas de mayor a menor diferencia entre centroides
plt.plot(dif) #Grafica de diferencias
plt.show()
 
   
#--------- Codo - Punto de Inflexión ---------------------------------

x=range(0,len(dif)) # Se crea lista con numeros consecutivos
data=np.asarray(list(zip(x,dif))) # Se crea tupla con x y dif
elbow_index= find_elbow(data, get_data_radiant(data)) # elbow_index contiene el punto de inflexion

#Grafica de diferencias, linea recta y el punto de inflexion
plt.plot(data[:, 0], data[:, 1])
plt.plot([data[:,0][0],data[:, 0][-1]],[data[:,1][0],data[:, 1][-1]])
plt.plot(data[elbow_index][0],data[elbow_index][1],'o')
print(elbow_index)

#------------ Guardar variables de interes ---------------------

corte=elbow_index

# Poblar el DF con las variables significativas
for k in var_orden[:corte]:
    bd_new[k[0]]=bd[k[0]]
    
bd_new=bd_new.drop(bd_new.columns[0], axis=1) # Se elimina l columna a, pues solo se utiliza para especificar el numero de registros
bd_new.to_excel('Base_Datos'+str(corte)+'variables.xlsx') # Se guarda excel con variables relevantes