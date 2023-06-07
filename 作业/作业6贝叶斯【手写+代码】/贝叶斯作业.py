#------------------------------------------------------read_data
import pandas as pd
df=pd.read_excel('PlayTennis训练样本_.xlsx')

#______________________________________________________________encode
from sklearn.preprocessing import LabelEncoder
#不需要解码X，但是y是需要解码的所以单独列出
def en_de_code_context(data,unco_context_cols):
    label=LabelEncoder()    
    for col in unco_context_cols:
        data[col]=label.fit_transform(data[col])
        print('encoded',col)
    return data
data=en_de_code_context(df.iloc[:,1:-1].copy(),df.columns.to_list()[1:-1])#给前4个属性直接编码
le=LabelEncoder()   
train_y=le.fit_transform(df['PlayTennis'].iloc[:-1])#给PlayTennis单独编码,
train_X=data.iloc[:-1,:]
test_X=data.iloc[[-1],:]

#___________________________________________________________________predict
from sklearn.naive_bayes import CategoricalNB
#拉普拉斯平滑默认，为1
cnb=CategoricalNB()
cnb.fit(train_X,train_y)
print('D15预测结果为：',le.inverse_transform(cnb.predict(test_X)))