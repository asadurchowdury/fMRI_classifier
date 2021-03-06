from os import replace
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering, OPTICS

# you can switch between the following two blocks for two task conditions
# hc_enc = pd.read_csv('assets/HC_encoding.csv',na_values='inf',header=None)
# scz_enc = pd.read_csv('assets/SCZ_encoding.csv',na_values='inf',header=None)
# final = pd.concat([hc_enc,scz_enc])
# final = shuffle(final)

hc_enc = pd.read_csv('assets/HC_retrieval.csv',na_values='inf',header=None)
scz_enc = pd.read_csv('assets/SCZ_retrieval.csv',na_values='inf',header=None)
final = pd.concat([hc_enc,scz_enc])
final = shuffle(final)

X_train = final.iloc[:,1:]
y_true = final.iloc[:,0]

# print(X_train.head())
# print(y_true)

# kmeans = AgglomerativeClustering(n_clusters=2, compute_distances=True,memory='./assets/',linkage='ward') 
# kmeans = KMeans(n_clusters=2,verbose=0,init='k-means++',tol=0.001,algorithm='full')
# kmeans = DBSCAN(eps=2,min_samples=2,metric='yule',n_jobs=-1,algorithm='brute')
optics = OPTICS(metric='canberra',cluster_method='xi',n_jobs=-1)
optics.fit_predict(X_train)

preds = optics.labels_


from sklearn.metrics import f1_score, accuracy_score
print(optics.labels_)
print(f1_score(y_true,preds))
print(accuracy_score(y_true,preds))