from os import replace
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans

hc_enc = pd.read_csv('assets/HC_zenc.csv',na_values='inf',header=None)
scz_enc = pd.read_csv('assets/SCZ_zenc.csv',na_values='inf',header=None)
# enc = hc_enc.conca
hc_enc = hc_enc.dropna(axis=1)
scz_enc = scz_enc.dropna(axis=1)
final = pd.concat([hc_enc,scz_enc])
print(hc_enc.head())
print(scz_enc.shape)


kmeans = KMeans(n_clusters=2,verbose=1,n_jobs=-1,init='k-means++',tol=0.001,algorithm='full')
kmeans.fit(final)

preds = kmeans.labels_
y_true = pd.read_csv('assets/labels.csv')

from sklearn.metrics import f1_score

print(f1_score(y_true,preds))

print(kmeans.labels_)