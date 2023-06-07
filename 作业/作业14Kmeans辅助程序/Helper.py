#!/usr/bin/python
# -*- coding: utf-8 -*-
# @author: GJH
# @file: Helper
# @time: 2023/4/24,20:11
import pandas as pd
from scipy.spatial import distance_matrix
def cal_AVG(index_list,df):
    return list(df.loc[index_list].mean())
def cal_distance_matrix(DF,U1,U2,U3):
    df=DF.copy()
    df.loc['U1'],df.loc['U2'],df.loc['U3']=U1,U2,U3
    matrix=pd.DataFrame(distance_matrix(df.values, df.values), index=df.index,columns=df.index)
    simple_matrix=matrix.loc[['U1','U2','U3']][list('ABCDEF')]
    print(simple_matrix)
points=[[0,0],[2,1],[2,-1],[2.1,0],[4.1,0],[5,0]]
df=pd.DataFrame(points,columns=['x','y'])
df.index=list('ABCDEF')