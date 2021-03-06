from os import replace
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering

hc_enc = pd.read_csv('assets/HC_encoding.csv',na_values='inf',header=None)
scz_enc = pd.read_csv('assets/SCZ_encoding.csv',na_values='inf',header=None)
# enc = hc_enc.conca
# hc_enc = hc_enc.dropna(axis=1)
# scz_enc = scz_enc.dropna(axis=1)
final = pd.concat([hc_enc,scz_enc])
final = shuffle(final)


X_train = final.iloc[:,1:]
y_true = final.iloc[:,0]

# print(X_train.head())
# print(y_true)

# kmeans = AgglomerativeClustering(n_clusters=2, compute_distances=True,memory='./assets/',linkage='ward') 
# kmeans = KMeans(n_clusters=2,verbose=0,init='k-means++',tol=0.001,algorithm='full')
# kmeans = DBSCAN(eps=2,min_samples=2,metric='yule',n_jobs=-1,algorithm='brute')
spectral = SpectralClustering(n_clusters=2)
spectral.fit_predict(X_train)

preds = spectral.labels_


from sklearn.metrics import f1_score, accuracy_score
print(spectral.labels_)
print(f1_score(y_true,preds))
print(accuracy_score(y_true,preds))