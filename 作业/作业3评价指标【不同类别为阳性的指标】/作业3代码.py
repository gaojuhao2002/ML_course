#-------------------------------------------------------------------------------------Init
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score
y_true = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
y_pred = [1, 0, 1, 1, 1, 1, 0, 0, 0, 0]
#-------------------------------------------------------------------------------------Q1
accuracy = accuracy_score(y_true, y_pred)
print("准确率：", accuracy)
#-------------------------------------------------------------------------------------Q2

def metric_appoint_pos(pos_label):
    recall = recall_score(y_true, y_pred, pos_label=pos_label)
    precision = precision_score(y_true, y_pred, pos_label=pos_label)
    f1 = f1_score(y_true, y_pred, pos_label=pos_label)
    print("{}为阳性时".format(pos_label),'-*'*20)
    print("召回率：", recall)
    print("精确率：", precision)
    print("F1值：", f1)
metric_appoint_pos(1)
metric_appoint_pos(0)

#-------------------------------------------------------------------------------------Q3

# 计算方式一：将2和3的结果直接平均
print(f"方式一计算得到的召回率：{recall_score(y_true,y_pred,average='macro')}")

# 计算方式二：将2和3的结果加权平均，权重由对应的阳性样本数决定
print(f"方式二计算得到的召回率：{recall_score(y_true,y_pred,average='weighted')}")

# 计算方式三
print(f"方式三计算得到的召回率：{recall_score(y_true, y_pred, average='micro')}")