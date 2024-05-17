# -*- coding: utf-8 -*-
"""
Created on Sat May  4 21:25:59 2024

@author: Asus
"""

#%%

"""
Hier werde ich versuchen, einen Code zu schreiben, der Clustering auf einem Datensatz durchführt, der das Alter der Kunden, ihr Geschlecht, ihr Jahreseinkommen und ihren Ausgabenwert enthält.
Ich werde versuchen, die Ausgabenbewertung in Gruppen zu gruppieren, die perfekte Anzahl benötigter Cluster zu finden und die Bereiche der Cluster zu sehen, damit wir diese Daten dann nutzen können, um Erkenntnisse zu gewinnen.

Quelle des Datensatzes für die Refrenzierung: https://www.kaggle.com/datasets/sinderpreet/customer-segmentation-and-clustering-python?resource=download, abgerufen am 05.07.2024

"""
#%%
#Hochladen der benötigten Bibliotheken für die Vorgänge
# Die Bibliotheken können mit Tabellen umgehen, Standardskalierung durchführen, Clustering mit Kmeans und Visualisierung durchführen
import pandas as pd
import sklearn.metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
#%%
# Laden Sie den Datensatz hoch und sehen Sie sich die ersten fünf Spalten an, um eine Vorstellung davon zu bekommen, was es enthält
df = pd.read_csv("C:\\Users\\Asus\\Desktop\\spyder cluster\\Mall_Customers.csv")
print(df.head())
#%%
# Ich werde Standardscaler verwenden, um die Konvergenz der Spalte und auch die gleiche Gewichtung der Werte zu erreichen.
# Da Standard-Skalierer davon ausgehen, dass die Daten in einem 2D-Array vorliegen, verwende ich „values.reshape(-1, 1)“ für die Spalte, um sie von 1D in 2D umzuwandeln.
SpendingScore = df['Spending Score (1-100)'].values.reshape(-1, 1)
scaler = StandardScaler()
spending_scores_scaled = scaler.fit_transform(SpendingScore)
#%%
# Hier verwende ich die Elbow-Methode, um die ideale Anzahl von Clustern zu erhalten
# Die beste Clusteranzahl wird erreicht, wenn die Trägheit nicht geringer wird
k_list = []
elbow_scores = []
for k in range(2, 51):
    kmeans_model = KMeans(n_clusters=k)
    kmeans_model.fit(spending_scores_scaled)
    elbow_scores.append(kmeans_model.inertia_)
    k_list.append(k)
plt.plot(k_list, elbow_scores)
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
#%%
# Die beste Zahl ist etwa 10, und jetzt werde ich die Spalte gruppieren
desired_number_of_clusters = 10
kmeans_model = KMeans(n_clusters=desired_number_of_clusters)
kmeans_model.fit(spending_scores_scaled)
#%%
# Ich werde die Cluster-Beschriftungen abrufen
# und sehen Sie sich dann die Trägheit der Spalte an, da Trägheit ein Maß für die Gruppierung von Clustern ist
cluster_labels = kmeans_model.labels_
print("Inertia:", kmeans_model.inertia_)
#%%
# Hier sehe ich die Cluster-Zentren, da sie uns den durchschnittlichen standardisierten Ausgabenwert für den EAHC-Cluster zeigen können und wie stark er vom durchschnittlichen Ausgabenwert abweicht
print("Cluster Centers:", kmeans_model.cluster_centers_)
#%%
# Hier füge ich eine Spalte hinzu, um die neu erstellten Cluster den Daten zuzuordnen, und sehe dann die ersten 5 Spalten
df['cluster_id'] = cluster_labels
print(df.head())
#%%
# Hier sehe ich den Maximal- und Minimalwert jedes Clusters
cluster_groups = df.groupby('cluster_id')
for cluster, group in cluster_groups:
    min_value = group['Spending Score (1-100)'].min()
    max_value = group['Spending Score (1-100)'].max()
    print(f'Cluster {cluster}: Min = {min_value}, Max = {max_value}')
#%%
# und hier werde ich Silhouette verwenden, um herauszufinden, wie nah die Motive an ihren eigenen Clustern sind
silhouette_score = sklearn.metrics.silhouette_score(spending_scores_scaled, cluster_labels)
print("Silhouette Score:", silhouette_score)
#%%
"""
Der Silhouetten-Score ist gut, aber es ist besser, wenn er nahe bei 1 liegt
Der Trägheitswert von 1,9 ist gut, wenn man den Bereich der Werte in jedem Cluster berücksichtigt
Die Ergebnisse der Clusterzentren sind gut, da sie nicht stark von der Mitte abweichen

Die Bandbreite der Cluster, die ich bekommen habe: Cluster 0: Min = 46, Max = 53
Cluster 1: Min = 88, Max = 99
Cluster 2: Min = 20, Max = 28
Cluster 3: Min = 66, Max = 78
Cluster 4: Min = 10, Max = 18
Cluster 5: Min = 54, Max = 65
Cluster 6: Min = 29, Max = 36
Cluster 7: Min = 39, Max = 45
Cluster 8: Min = 1, Max = 9
Cluster 9: Min = 79, Max = 87

Jeder Cluster kann einen Unterschied zwischen 11 und 8 zwischen seinen Maximal- und Minimalwerten aufweisen. Liegen die Bereiche nahe beieinander, kann das bedeuten, dass sie fair sind
"""