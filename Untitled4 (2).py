
# coding: utf-8

# In[220]:


import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend as K
from pylab import *
from sklearn import linear_model
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


# In[2]:


data_fraud = pd.read_csv('table_1_join_2.csv', sep=',')
data_fraud = data_fraud[['app_id','flag']]
data_fraud['app_id'] = list(map(int, data_fraud['app_id'].map(lambda x: str(x)[2:])))
data_fraud = data_fraud.sort_values(by=['app_id'])


# In[3]:


data1 = pd.read_csv('table_6_1.csv', sep=',')
data1['app_id'] = list(map(int, data1['app_id'].map(lambda x: str(x)[2:])))
data1 = data1.loc[:,data1.sum() != 0]
zero_data = data1[:1]*0
for i in range(len(data_fraud)):
    if len(data1.loc[data1['app_id'] == data_fraud['app_id'][i:i+1].as_matrix()[0]]) == 0:
        data1 = data1.append(zero_data)
        data1['app_id'][-1:] = data_fraud['app_id'][i:i+1].as_matrix()[0]
        print((len(data1) - 45878)/(50500 - 45878)*100, '%')
data1 = data1.sort_values(by=['app_id'])
data1


# In[4]:


data2 = pd.read_csv('table_6_2.csv', sep=',')
data2['app_id'] = list(map(int, data2['app_id'].map(lambda x: str(x)[2:])))
data2 = data2.loc[:,data2.sum() != 0]
data2.columns = data2.columns[0:1].append('2' + data2.columns[1:])
zero_data = data2[:1]*0
for i in range(len(data_fraud)):
    if len(data2.loc[data2['app_id'] == data_fraud['app_id'][i:i+1].as_matrix()[0]]) == 0:
        data2 = data2.append(zero_data)
        data2['app_id'][-1:] = data_fraud['app_id'][i:i+1].as_matrix()[0]
        print((len(data2) - 39415)/(50500 - 39415)*100, '%')
data2 = data2.sort_values(by=['app_id'])
data2


# In[5]:


data3 = pd.read_csv('table_6_3.csv', sep=',')
data3['app_id'] = list(map(int, data3['app_id'].map(lambda x: str(x)[2:])))
data3 = data3.loc[:,data3.sum() != 0]
data3.columns = data3.columns[0:1].append('3' + data3.columns[1:])
zero_data = data3[:1]*0
for i in range(len(data_fraud)):
    if len(data3.loc[data3['app_id'] == data_fraud['app_id'][i:i+1].as_matrix()[0]]) == 0:
        data3 = data3.append(zero_data)
        data3['app_id'][-1:] = data_fraud['app_id'][i:i+1].as_matrix()[0]
        print((len(data3) - 8274)/(50500 - 8274)*100, '%')
data3 = data3.sort_values(by=['app_id'])
data3


# In[11]:


data_fraud = pd.read_csv('table_1_join_2.csv', sep=',')
data_fraud = data_fraud[['app_id', 'flag']]
data_fraud['app_id'] = list(map(int, data_fraud['app_id'].map(lambda x: str(x)[2:])))
#data_fraud = data_fraud.sort_values(by=['app_id'])
dataset = data_fraud.merge(data1, on="app_id").merge(data2, on="app_id").merge(data3, on="app_id")
#for i in range(72):
#    dataset[dataset.columns[i+2]] = pd.DataFrame(dataset[dataset.columns[i+2]] + dataset[dataset.columns[i+157]])
#for i in range(83):
#    dataset[dataset.columns[i+2]] = pd.DataFrame(dataset[dataset.columns[i+2]] + dataset[dataset.columns[i+229]])
#dataset[dataset.columns[1:157]]
dataset = dataset.drop('app_id', axis=1)
dataset_1 = dataset[:500].append(dataset[:500]).append(dataset[:500]).append(dataset[:500]).append(dataset[:500]).sample(frac=1)
dataset_0 = dataset[500:].sample(frac=1)
dataset_1


# In[9]:


#for i in range(50):
#    dataset_0 = dataset_0.append(dataset_1)
#dataset_0 = dataset_0.sample(frac=1)
#dataset_0


# In[12]:


test = dataset_1[:50].append(dataset_0[:950])
testY = test[["flag"]]
testX = test.drop('flag', axis=1)
testY


# In[13]:


train = dataset_0[950:].append(dataset_1[50:])
train = train.sample(frac=1)
trainY = train[["flag"]]
trainX = train.drop('flag', axis=1)
trainY


# In[14]:


len(testY.loc[testY['flag']==1])


# In[15]:


def f1(y_true, y_pred):
    precision1 = precision(y_true, y_pred)
    recall1 = recall(y_true, y_pred)
    return 2*((precision1*recall1)/(precision1+recall1+K.epsilon()))

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall1 = true_positives / (possible_positives + K.epsilon())
    print(true_positives, possible_positives, end=' ')
    return recall1

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision1 = true_positives / (predicted_positives + K.epsilon())
    print(true_positives, predicted_positives)
    return precision1


# In[16]:


logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(trainX, trainY)


# In[20]:


pred1 = logreg.predict(testX)
pred1 = pd.DataFrame(pred)
pred1.columns = ['flag']
new1 = pd.DataFrame(np.hstack((pred1, testY)))
new1.columns = ['p', 't']
new1


# In[21]:


tp1 = len(new1.loc[new1['p']==1].loc[new1.loc[new1['p']==1]['t']==1])
tn1 = len(new1.loc[new1['p']==0].loc[new1.loc[new1['p']==0]['t']==0])
fn1 = len(new1.loc[new1['p']==0].loc[new1.loc[new1['p']==0]['t']==1])
fp1 = len(new1.loc[new1['p']==1].loc[new1.loc[new1['p']==1]['t']==0])
print(tp1, fn1, '\n', fp1, tn1)
pr1 = tp1 / (tp1 + fp1)
re1 = tp1 / (tp1 + fn1)
print('precision = ', pr1, '\n recall = ', re1, '\n f = ', (2 * pr1 * re1 / (pr1 + re1)))


# In[24]:


model1=Sequential()
model1.add(Dense(input_dim=trainX.shape[1],units=300, activation="sigmoid"))
#model1.add(Dropout(0.25))
#model1.add(Dense(input_dim=trainX.shape[1],units=50, activation="sigmoid"))
#model1.add(Dropout(0.25))
model1.add(Dense(1, activation="sigmoid"))
model1.compile(optimizer='Adam', loss="binary_crossentropy", metrics=[f1])


# In[44]:


model1.fit(trainX,trainY, epochs=30, verbose=2, batch_size=100)
model1.save_weights("trg-f.h5")


# In[46]:


pred2 = model1.predict(testX).round().astype(int)
pred2 = pd.DataFrame(pred2)
pred2.columns = ['flag']
new2 = pd.DataFrame(np.hstack((pred2, testY)))
new2.columns = ['p', 't']
new2


# In[47]:


tp2 = len(new2.loc[new2['p']==1].loc[new2.loc[new2['p']==1]['t']==1])
tn2 = len(new2.loc[new2['p']==0].loc[new2.loc[new2['p']==0]['t']==0])
fn2 = len(new2.loc[new2['p']==0].loc[new2.loc[new2['p']==0]['t']==1])
fp2 = len(new2.loc[new2['p']==1].loc[new2.loc[new2['p']==1]['t']==0])
print(tp2, fn2, '\n', fp2, tn2)
pr2 = tp2 / (tp2 + fp2)
re2 = tp2 / (tp2 + fn2)
print('precision = ', pr2, '\n recall = ', re2, '\n f = ', (2 * pr2 * re2 / (pr2 + re2)))


# In[57]:


model2=Sequential()
model2.add(Dense(input_dim=trainX.shape[1],units=300, activation="sigmoid"))
model2.add(Dropout(0.25))
model2.add(Dense(input_dim=trainX.shape[1],units=50, activation="sigmoid"))
model2.add(Dropout(0.25))
model2.add(Dense(1, activation="sigmoid"))
model2.compile(optimizer='Adam', loss="binary_crossentropy", metrics=[f1])


# In[82]:


model2.fit(trainX,trainY, epochs=30, verbose=2, batch_size=100)
model2.save_weights("trg-f.h5")


# In[83]:


pred3 = model2.predict(testX).round().astype(int)
pred3 = pd.DataFrame(pred3)
pred3.columns = ['flag']
new3 = pd.DataFrame(np.hstack((pred3, testY)))
new3.columns = ['p', 't']
new3


# In[84]:


tp3 = len(new3.loc[new3['p']==1].loc[new3.loc[new3['p']==1]['t']==1])
tn3 = len(new3.loc[new3['p']==0].loc[new3.loc[new3['p']==0]['t']==0])
fn3 = len(new3.loc[new3['p']==0].loc[new3.loc[new3['p']==0]['t']==1])
fp3 = len(new3.loc[new3['p']==1].loc[new3.loc[new3['p']==1]['t']==0])
print(tp3, fn3, '\n', fp3, tn3)
pr3 = tp3 / (tp3 + fp3)
re3 = tp3 / (tp3 + fn3)
print('precision = ', pr3, '\n recall = ', re3, '\n f = ', (2 * pr3 * re3 / (pr3 + re3)))


# In[85]:


model3=Sequential()
model3.add(Dense(input_dim=trainX.shape[1],units=300, activation="sigmoid"))
model3.add(Dropout(0.5))
model3.add(Dense(input_dim=trainX.shape[1],units=50, activation="sigmoid"))
model3.add(Dropout(0.5))
model3.add(Dense(1, activation="sigmoid"))
model3.compile(optimizer='Adam', loss="binary_crossentropy", metrics=[f1])


# In[104]:


model3.fit(trainX,trainY, epochs=30, verbose=2, batch_size=100)
model3.save_weights("trg-f.h5")


# In[105]:


pred4 = model3.predict(testX).round().astype(int)
pred4 = pd.DataFrame(pred4)
pred4.columns = ['flag']
new4 = pd.DataFrame(np.hstack((pred4, testY)))
new4.columns = ['p', 't']
new4


# In[106]:


tp4 = len(new4.loc[new4['p']==1].loc[new4.loc[new4['p']==1]['t']==1])
tn4 = len(new4.loc[new4['p']==0].loc[new4.loc[new4['p']==0]['t']==0])
fn4 = len(new4.loc[new4['p']==0].loc[new4.loc[new4['p']==0]['t']==1])
fp4 = len(new4.loc[new4['p']==1].loc[new4.loc[new4['p']==1]['t']==0])
print(tp4, fn4, '\n', fp4, tn4)
pr4 = tp4 / (tp4 + fp4)
re4 = tp4 / (tp4 + fn4)
print('precision = ', pr4, '\n recall = ', re4, '\n f = ', (2 * pr4 * re4 / (pr4 + re4)))


# In[112]:


model4=Sequential()
model4.add(Dense(input_dim=trainX.shape[1],units=30, activation="sigmoid"))
#model4.add(Dropout(0.2))
#model4.add(Dense(input_dim=trainX.shape[1],units=10, activation="sigmoid"))
#model4.add(Dropout(0.2))
model4.add(Dense(1, activation="sigmoid"))
model4.compile(optimizer='Adam', loss="binary_crossentropy", metrics=[f1])


# In[133]:


model4.fit(trainX,trainY, epochs=30, verbose=2, batch_size=100)
model4.save_weights("trg-f.h5")


# In[134]:


pred5 = model4.predict(testX).round().astype(int)
pred5 = pd.DataFrame(pred5)
pred5.columns = ['flag']
new5 = pd.DataFrame(np.hstack((pred5, testY)))
new5.columns = ['p', 't']
new5


# In[135]:


tp5 = len(new5.loc[new5['p']==1].loc[new5.loc[new5['p']==1]['t']==1])
tn5 = len(new5.loc[new5['p']==0].loc[new5.loc[new5['p']==0]['t']==0])
fn5 = len(new5.loc[new5['p']==0].loc[new5.loc[new5['p']==0]['t']==1])
fp5 = len(new5.loc[new5['p']==1].loc[new5.loc[new5['p']==1]['t']==0])
print(tp5, fn5, '\n', fp5, tn5)
pr5 = tp5 / (tp5 + fp5)
re5 = tp5 / (tp5 + fn5)
print('precision = ', pr5, '\n recall = ', re5, '\n f = ', (2 * pr5 * re5 / (pr5 + re5)))


# In[136]:


model5=Sequential()
model5.add(Dense(input_dim=trainX.shape[1],units=500, activation="sigmoid"))
model4.add(Dropout(0.1))
model4.add(Dense(input_dim=trainX.shape[1],units=300, activation="sigmoid"))
model4.add(Dropout(0.1))
model4.add(Dense(input_dim=trainX.shape[1],units=100, activation="sigmoid"))
model4.add(Dropout(0.1))
model5.add(Dense(1, activation="sigmoid"))
model5.compile(optimizer='Adam', loss="binary_crossentropy", metrics=[f1])


# In[157]:


model5.fit(trainX,trainY, epochs=30, verbose=2, batch_size=100)
model5.save_weights("trg-f.h5")


# In[158]:


pred6 = model5.predict(testX).round().astype(int)
pred6 = pd.DataFrame(pred6)
pred6.columns = ['flag']
new6 = pd.DataFrame(np.hstack((pred6, testY)))
new6.columns = ['p', 't']
new6


# In[159]:


tp6 = len(new6.loc[new6['p']==1].loc[new6.loc[new6['p']==1]['t']==1])
tn6 = len(new6.loc[new6['p']==0].loc[new6.loc[new6['p']==0]['t']==0])
fn6 = len(new6.loc[new6['p']==0].loc[new6.loc[new6['p']==0]['t']==1])
fp6 = len(new6.loc[new6['p']==1].loc[new6.loc[new6['p']==1]['t']==0])
print(tp6, fn6, '\n', fp6, tn6)
pr6 = tp6 / (tp6 + fp6)
re6 = tp6 / (tp6 + fn6)
print('precision = ', pr6, '\n recall = ', re6, '\n f = ', (2 * pr6 * re6 / (pr6 + re6)))


# In[210]:


tree = DecisionTreeClassifier(max_depth=100, random_state=17)


# In[211]:


tree.fit(trainX, trainY)


# In[212]:


pred7 = tree.predict(testX).round().astype(int)
pred7 = pd.DataFrame(pred7)
pred7.columns = ['flag']
new7 = pd.DataFrame(np.hstack((pred7, testY)))
new7.columns = ['p', 't']
new7


# In[213]:


tp7 = len(new7.loc[new7['p']==1].loc[new7.loc[new7['p']==1]['t']==1])
tn7 = len(new7.loc[new7['p']==0].loc[new7.loc[new7['p']==0]['t']==0])
fn7 = len(new7.loc[new7['p']==0].loc[new7.loc[new7['p']==0]['t']==1])
fp7 = len(new7.loc[new7['p']==1].loc[new7.loc[new7['p']==1]['t']==0])
print(tp7, fn7, '\n', fp7, tn7)
pr7 = tp7 / (tp7 + fp7)
re7 = tp7 / (tp7 + fn7)
print('precision = ', pr7, '\n recall = ', re7, '\n f = ', (2 * pr7 * re7 / (pr7 + re7)))


# In[246]:


knn = KNeighborsClassifier(n_neighbors=2500)


# In[247]:


knn.fit(trainX, trainY)


# In[248]:


pred8 = knn.predict(testX).round().astype(int)
pred8 = pd.DataFrame(pred8)
pred8.columns = ['flag']
new8 = pd.DataFrame(np.hstack((pred8, testY)))
new8.columns = ['p', 't']
new8


# In[249]:


tp8 = len(new8.loc[new8['p']==1].loc[new8.loc[new8['p']==1]['t']==1])
tn8 = len(new8.loc[new8['p']==0].loc[new8.loc[new8['p']==0]['t']==0])
fn8 = len(new8.loc[new8['p']==0].loc[new8.loc[new8['p']==0]['t']==1])
fp8 = len(new8.loc[new8['p']==1].loc[new8.loc[new8['p']==1]['t']==0])
print(tp8, fn8, '\n', fp8, tn8)
pr8 = tp8 / (tp8 + fp8)
re8 = tp8 / (tp8 + fn8)
print('precision = ', pr8, '\n recall = ', re8, '\n f = ', (2 * pr8 * re8 / (pr8 + re8)))


# In[283]:


clf = RandomForestClassifier(n_estimators=50, max_depth=50, random_state=0)


# In[284]:


clf.fit(trainX, trainY)


# In[285]:


pred9 = clf.predict(testX).round().astype(int)
pred9 = pd.DataFrame(pred9)
pred9.columns = ['flag']
new9 = pd.DataFrame(np.hstack((pred9, testY)))
new9.columns = ['p', 't']
new9


# In[286]:


tp9 = len(new9.loc[new9['p']==1].loc[new9.loc[new9['p']==1]['t']==1])
tn9 = len(new9.loc[new9['p']==0].loc[new9.loc[new9['p']==0]['t']==0])
fn9 = len(new9.loc[new9['p']==0].loc[new9.loc[new9['p']==0]['t']==1])
fp9 = len(new9.loc[new9['p']==1].loc[new9.loc[new9['p']==1]['t']==0])
print(tp9, fn9, '\n', fp9, tn9)
pr9 = tp9 / (tp9 + fp9)
re9 = tp9 / (tp9 + fn9)
print('precision = ', pr9, '\n recall = ', re9, '\n f = ', (2 * pr9 * re9 / (pr9 + re9)))

