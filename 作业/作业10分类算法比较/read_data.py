#!/usr/bin/python
# -*- coding: utf-8 -*-
# @author: GJH
# @file: read_data

import scipy.io as scio
import numpy as np


def data_oeprator(data):
    #自定义对数据的操作例如取平均值
    return np.mean(data)

def group_vgg(data_dict):#由于未知数据类型，例如对行或列可以自行加入axis等
    return {k:np.mean(v) for k,v in data_dict.items()}

def get_data(path_,key):
    data_dict = scio.loadmat(path_)#读出来是一个字典
    return data_dict[key] #返回想要的部分数据
def Treavers(path):
    year = [x for x in range(2008, 2013)]
    month = [x for x in range(6, 9)]
    data_dict = dict(zip(year, {i: [] for i in month}))

    for y in year:
        for m in month:
            str_y=str(y) if y>=10 else '0'+str(y)
            str_m='0'+str(m)

            len_day=30 if m == 6 else 31  #七月八月31天，6月30天
            day_range=[x for x in range(1,len_day+1)]

            for d in day_range:
                str_d=str(d) if d>=10 else '0'+str(d)
                path_=path+'/'+str_y+str_m+str_d+'.mat'

                data=get_data(path_,'A')#例如读取mat文件里面的A数据。或传列表
                data=data_oeprator(data)#聚合

                data_dict[y][m].appen(data)#加入聚合后数据
                # print(path_)
    return data_dict

def count_per_month(data_dict):#类似拉伸的
    data_per_month={}
    for year,v in data_dict.items():
        for month,day_avg_data in v.items():
            str_y=str(year) if year>=10 else '0'+str(year)
            str_m='0'+str(month)
            data_per_month[str_y+str_m]=day_avg_data
    return data_per_month
def count_3or5_month(data_per_month,type_):
    #type_是选择3或5
    #下面这个循环分成每type_月一组
    res={}
    temp = []
    for ind,(k,v) in enumerate(data_per_month.items(),1):
        start_data= k if ind //type_ else None#开始的日期，或者定义为其他的日期，看需求也可以不要
        temp.append(v)
        if len(temp)==type_:#达到对应组长度
            res[start_data]=temp
            temp=[]# 清空，获取下一组
    return res





if __name__ =='__main__':
    data_dict=Treavers('www')
    data_per_month=count_per_month(data_dict.copy())

    month_1=group_vgg(data_per_month.copy())
    month_3=group_vgg(count_3or5_month(data_per_month,3))
    month_5=group_vgg(count_3or5_month(data_per_month,5))