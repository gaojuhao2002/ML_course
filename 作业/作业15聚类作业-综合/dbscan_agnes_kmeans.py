#!/usr/bin/python
# -*- coding: utf-8 -*-
# @author: GJH
# @file: dbscan_agnes_kmeans
# @time: 2023/5/9,14:59
from sklearn.metrics import silhouette_score,silhouette_samples

from sklearn.cluster import DBSCAN,AgglomerativeClustering,KMeans
import numpy as np


if __name__=='__main__':
    X=np.array([[0,0],[0,1],[3,1],[3,0],[5,0], [5,1]])
    clser=[
        DBSCAN(eps=1.1, min_samples=2),
        AgglomerativeClustering(n_clusters=3, linkage='single'),
        KMeans(n_clusters=3, init=np.array([[0, 0], [0, 1], [3, 1]]))
           ]
    names=['DBSCAN','AGENS','KMEANS']
    for name,cls in zip(names,clser):
        cls.fit(X)
        res=cls.labels_
        print(name,'聚类结果:',res,'轮廓系数',silhouette_score(X, res))
        print('每个样本的轮廓系数',silhouette_samples(X,res))