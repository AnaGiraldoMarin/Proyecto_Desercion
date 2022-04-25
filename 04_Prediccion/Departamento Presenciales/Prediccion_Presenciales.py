# -*- coding: utf-8 -*-
"""

@author: AnaMaria
"""

#----------- Librerias Utlizadas ------------------------

import pandas as pd
from math import dist

print('Modelo - Todo el departamento') # Para informar al usuario el modelo que esta corriendo


#----------------- Inicializacion -------------------------

#DataFrames
centroides=pd.read_excel("centroids12.xlsx",index_col=0) #DF con la los valores de los centroides
estu= pd.read_excel("EstudiantePresenciales.xlsx",index_col=0) #DF con la informacion del estudiante

#Listas
Distancia_a_Centroides=[]
Orden_Pertenecia=[]

#Bandera
bandera = 0
#---------------- Calculo Distancias -----------------------------
for k in range(0,centroides.shape[0]):
    Distancia_a_Centroides.append([dist(centroides.iloc[k],estu.iloc[0]), k])
    
#------------ Ordenar las distancias de menor a mayor -------------
for k in range(0,centroides.shape[0]):
    Orden_Pertenecia.append(Distancia_a_Centroides[Distancia_a_Centroides.index(min(Distancia_a_Centroides))][-1])
    if (4 in Orden_Pertenecia or 8 in Orden_Pertenecia or 6 in Orden_Pertenecia or 3 in Orden_Pertenecia )   & bandera == 0:
        bandera=1;
        Perfil = Distancia_a_Centroides[Distancia_a_Centroides.index(min(Distancia_a_Centroides))][-1];
    del Distancia_a_Centroides[Distancia_a_Centroides.index(min(Distancia_a_Centroides))]
    
print('El estudiante tiene una mayor probabilidad de pertenecia a estos clusters: '+str(Orden_Pertenecia[:5])+
      ' de los perfiles de desercion el mas cercano es '+str(Perfil))