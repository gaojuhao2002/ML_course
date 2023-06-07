# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 08:24:11 2022

@author: dx.z
"""

from read_data import read_from_txt
import numpy as np

# 读取训练集
X_train_1024, X_train_64, Y_train = read_from_txt('./UCI_digits.train', Dim1 = 1024, Dim2 = 64)
# 读取测试集
X_test_1024, X_test_64, Y_test =  read_from_txt('./UCI_digits.test', Dim1 = 1024, Dim2 = 64)

# 把数据合在一起
X_1024 = np.concatenate((X_train_1024, X_test_1024))
X_64 = np.concatenate((X_train_64, X_test_64))
Y = np.concatenate((Y_train, Y_test))