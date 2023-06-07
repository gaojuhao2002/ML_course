# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 14:28:47 2021

@author: dx.z
"""
import numpy as np
import os

# 将1024维特征降维到64
def reduce_dim(feature, Dim = 64):
    X = feature.reshape((32, 32))
    x = np.zeros(Dim)
    for i_r in range(8):
        for i_c in range(8):
            patch = X[i_r*4:i_r*4+4, i_c*4:i_c*4+4]
            x[i_r*8+i_c] = patch.sum()
    return x

# 从文本文件读取1024维特征
# 同时返回降维后的特征及标签
def read_from_txt(file_name, Dim1 = 1024, Dim2 = 64):
    fid = open(file_name, 'r')
    datas = fid.readlines()
    X_1024 = np.zeros((len(datas), Dim1), dtype=np.float64)
    X_64 = np.zeros((len(datas), Dim2), dtype=np.float64)
    Y = np.zeros(len(datas), dtype=np.uint8)
    for idx, data in enumerate(datas):
        img_1024 = [np.float64(d) for d in data[:Dim1]]
        img_1024 = np.array(img_1024)
        X_1024[idx, :] = img_1024
        X_64[idx, :] = reduce_dim(img_1024)
        Y[idx] = np.uint8(data[Dim1])
    fid.close()
    return X_1024, X_64, Y