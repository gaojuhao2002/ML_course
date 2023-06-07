#!/usr/bin/python
# -*- coding: utf-8 -*-
# @author: GJH
# @file: 回归树-KNN
# @time: 2023/4/18,14:41
x=[[x] for x in range(1,11)]
y=[[y] for y in [5.56,5.70,5.91,6.40,6.80,7.05,8.90,8.70,9.00,9.05]]
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
dct=DecisionTreeRegressor(max_depth=2)
knn=KNeighborsRegressor(n_neighbors=3)
dct.fit(x,y)
knn.fit(x,y)
print('dct:x={}'.format(dct.predict([[4.2]])))
print('knn:x={}'.format(knn.predict([[4.2]])))