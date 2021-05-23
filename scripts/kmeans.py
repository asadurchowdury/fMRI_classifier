from os import replace
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans

hc_enc = pd.read_csv('assets/HC_zenc.csv',na_values='inf',header=None)
scz_enc = pd.read_csv('assets/SCZ_zenc.csv',na_values='inf',header=None)
# enc = hc_enc.conca
hc_enc = hc_enc.dropna(axis=1)
scz_enc = scz_enc.dropna(axis=1)
final = pd.concat(hc_enc,scz_enc)
print(hc_enc.head())
print(hc_enc.shape)


kmeans = KMeans(n_clusters=2,verbose=1,n_jobs=-1)
kmeans.fit(final)

print(kmeans.labels_)