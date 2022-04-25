# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 8:24:22 2021

@author: AnaMaria
"""

#----------- Librerias Utlizadas ------------------------

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,davies_bouldin_score
from sklearn.metrics import pairwise_distances
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


#---------------- Funciones ---------------------------

#Funcion para graficar las distribuciones de desertores en cada cluster
def graficar(posicion,cluster,col):
        plt.subplot(5,5,posicion) # Se espeficica tamaño de subplot
        im=cluster[col].value_counts().sort_index() #Se ordena los valores de menor a mayor
        im.plot(kind='pie',
                autopct='%1.1f%%',
                fontsize=8,
                figsize=(6,6))
        plt.ylabel('')
        return((im/im.sum()).tolist()) # Retorna porcentajes de la distribucion


#----------------- Inicializacion -------------------------

#Creacion de carpetas para almacenar la informacion obtenida
os.mkdir('img') 
os.mkdir('excel') 

#DataFrames
bd=pd.read_excel("BaseDatos_Normalizada.xlsx",index_col=0) #DF con la BD normalizada

#Listas
Silhouette=[] # Lista donde se almacenaran los valores del indice Silouette
DaviesB=[] # Lista donde se almacenaran los valores del indice Davies Bouldin 
Dunn=[] # Lista donde se almacenaran los valores del indice Dunn
Proporciones=[] #Lista donde se almacena la mejor proporcion entre desertores y no desertores para cada numero de clusters
orden_proporcion=[] #Lista donde se almacenan las 4 mejores agrupaciones d eacuerdo a proporciones

#Variables
num_max_clusters=25  #Numero maximo de clsutars a analizar
agrupaciones_significativas= 3 #Numero de agrupciones mas significativas de acuerdo a distribucion de desertores en clusters

for k in range(2,num_max_clusters+1):
       
    
#-------------------- Algoritmo KMeans ------------------    
    kmeans = KMeans(n_clusters=k,verbose=False,tol=1e-6,n_init=600).fit(bd)
    labels=kmeans.labels_
    centroids = kmeans.cluster_centers_
   
    
#-------------------- Indices -----------------------------

#Silhouette
    Silhouette.append(silhouette_score(bd,labels))

#Davies Bouldin
    DaviesB.append(davies_bouldin_score(bd,labels))
    
#Dunn 
    n_labels=k
    intra_dists = np.zeros(n_labels)
    inter_dists = np.zeros(n_labels)
    centroid_dunn = np.zeros((n_labels, len(centroids[0])), dtype=np.float)
    for i in range(n_labels):
        cluster_k = bd[labels == i]
        centroid_dunn[i] = np.mean(cluster_k, axis=0)
        intra_dists[i] = np.max(pairwise_distances(cluster_k, [centroid_dunn[i]]))
    dis=pairwise_distances(centroid_dunn)
    for i in range(len(dis)):
        dis[i,i]=np.amax(dis)
    inter_dists=dis
    indice_dunn=(0. if np.max(intra_dists) == 0. else np.min(inter_dists)/np.max(intra_dists))
    Dunn.append(indice_dunn)
    
    
#------------------ Almacenamiento en archivos -----------

    data_df = pd.DataFrame(centroids)
    data_df.to_excel('excel/centroids'+str(k)+'.xlsx')
    data_df = pd.DataFrame(labels)
    data_df.to_excel('excel/labels'+str(k)+'.xlsx')

    
#------------------- Grafica Pie --------------
    Prop_internas=[] #Lista donde se almacenan las proporciones entre desertores y no desertores
    a1=pd.read_excel("Desertores.xlsx") #DF con la informacion de los estudiantes (Desertores o No Desertores) 
    b1=pd.read_excel('excel/labels'+str(k)+'.xlsx') # DF con la informacion sobre el label asignado a acada estudiante
    ab=a1
    ab["CLUSTER"]=b1[0] # Se agrupa la informacion de los estudiantes, estado de desercion y label asignado
    groups = ab.groupby(ab.CLUSTER) #Se agrupa por labels
    
    #Se crea DF con cada label
    c0 = groups.get_group(0) 
    if 1 in ab.values:
        c1=groups.get_group(1)
    if 2 in ab.values:
        c2=groups.get_group(2)
    if 3 in ab.values:
        c3=groups.get_group(3)
    if 4 in ab.values:
        c4=groups.get_group(4)
    if 5 in ab.values:
        c5=groups.get_group(5)
    if 6 in ab.values:
        c6=groups.get_group(6)
    if 7 in ab.values:
        c7=groups.get_group(7)
    if 8 in ab.values:
        c8=groups.get_group(8)
    if 9 in ab.values:
        c9=groups.get_group(9)
    if 10 in ab.values:
        c10=groups.get_group(10)
    if 11 in ab.values:
        c11=groups.get_group(11)
    if 12 in ab.values:
        c12=groups.get_group(12)
    if 13 in ab.values:
        c13=groups.get_group(13)
    if 14 in ab.values:
        c14=groups.get_group(14)
    if 15 in ab.values:
        c15=groups.get_group(15)
    if 16 in ab.values:
        c16=groups.get_group(16)
    if 17 in ab.values:
        c17=groups.get_group(17)
    if 18 in ab.values:
        c18=groups.get_group(18)
    if 19 in ab.values:
        c19=groups.get_group(19)
    if 20 in ab.values:
        c20=groups.get_group(20)
    if 21 in ab.values:
        c21=groups.get_group(21)
    if 22 in ab.values:
        c22=groups.get_group(22)
    if 23 in ab.values:
        c23=groups.get_group(23)
    if 24 in ab.values:
        c24=groups.get_group(24)
        
    #Se grafica la proporcion de estudiantes desertores y no desertores pertenecientes a cada cluster
    col="DESERTOR"
    print(k,col)
    Prop_internas.append(graficar(1,c0,col))
    plt.title(col+' '+str(k))
    if 'c1' in locals():
        Prop_internas.append(graficar(2,c1,col))
    if 'c2' in locals():
        Prop_internas.append(graficar(3,c2,col))
    if 'c3' in locals():
        Prop_internas.append(graficar(4,c3,col))
    if 'c4' in locals():
        Prop_internas.append(graficar(5,c4,col))
    if 'c5' in locals():
        Prop_internas.append(graficar(6,c5,col))
    if 'c6' in locals():
        Prop_internas.append(graficar(7,c6,col))
    if 'c7' in locals():
        Prop_internas.append(graficar(8,c7,col))
    if 'c8' in locals():
        Prop_internas.append(graficar(9,c8,col))
    if 'c9' in locals():
        Prop_internas.append(graficar(10,c9,col))
    if 'c10' in locals():
        Prop_internas.append(graficar(11,c10,col))
    if 'c11' in locals():
        Prop_internas.append(graficar(12,c11,col))
    if 'c12' in locals():
        Prop_internas.append(graficar(13,c12,col))
    if 'c13' in locals():
        Prop_internas.append(graficar(14,c13,col))
    if 'c14' in locals():
        Prop_internas.append(graficar(15,c14,col))
    if 'c15' in locals():
        Prop_internas.append(graficar(16,c15,col))
    if 'c16' in locals():
        Prop_internas.append(graficar(17,c16,col))
    if 'c17' in locals():
        Prop_internas.append(graficar(18,c17,col))
    if 'c18' in locals():
        Prop_internas.append(graficar(19,c18,col))
    if 'c19' in locals():
        Prop_internas.append(graficar(20,c19,col))
    if 'c20' in locals():
        Prop_internas.append(graficar(21,c20,col))
    if 'c21' in locals():
        Prop_internas.append(graficar(22,c21,col))
    if 'c22' in locals():
        Prop_internas.append(graficar(23,c22,col))
    if 'c23' in locals():
        Prop_internas.append(graficar(24,c23,col))
    if 'c24' in locals():
        Prop_internas.append(graficar(25,c24,col))
        
    plt.savefig('img/'+str(col)+str(k)+'.png')
    plt.show()
    
    #Analisis de proporciones para cada cluster
    Prop_internas.append(k)
    Proporciones.append([max(Prop_internas[:-1]), Prop_internas[-1]])
 
    
#---------------Analisis de proporciones------------------------------
for k in range(0,agrupaciones_significativas):
    orden_proporcion.append(Proporciones[Proporciones.index(max(Proporciones))][-1])
    del Proporciones[Proporciones.index(max(Proporciones))]

print ('Las'+str(agrupaciones_significativas)+' agrupaciones con la proporcion mas discriminativa en orden son: '+str(orden_proporcion))

#-----------------Grafica de indices ------------------------------

k=range(2,num_max_clusters+1) # Se crea vector

#Se normalizan los indices
a=Silhouette
norm_S = [(float(i)-min(a))/(max(a)-min(a)) for i in a]
a=DaviesB
norm_D = [(float(i)-min(a))/(max(a)-min(a)) for i in a]
a=Dunn
norm_Dunn = [Dunn if max(Dunn)==0. else 
             [(float(i)-min(a))/(max(a)-min(a)) for i in a]][0]

#Se grafica indice Silouette
plt.plot(k,norm_S,'o-',label="Silhouette (Max)")
plt.xlabel("Numero de Clusters")
plt.ylabel("Valor de Indice Normalizado")
plt.xticks(k,k)
plt.legend(fontsize=8)
plt.savefig('Silhouette.png')
plt.show()

#Se grafican Indices Davies Bouldin y Dunn 
plt.plot(k,norm_D,'o-',label="Davies (Min)")
plt.plot(k,norm_Dunn,'o-',label="Dunn (Max)")
plt.xlabel("Numero de Clusters")
plt.ylabel("Valor de Indice Normalizado")
plt.xticks(k,k)
plt.legend(fontsize=8)
plt.savefig('IndicesDavies_y_Dunn.png')