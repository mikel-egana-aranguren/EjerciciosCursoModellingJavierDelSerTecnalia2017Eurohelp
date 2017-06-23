# -*- coding: utf-8 -*-
"""
Created on Thu May 18 22:05:39 2017

@author: 106380
"""

import numpy as np
import requests
import json
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix

IP_SERVER = '10.42.0.1:5000'
NAME_OF_GROUP = 'MasterOfPuppets'
NAME_OF_MODEL = 'RandomForest-test'

######################################################

X_train = np.load(open('X_train.npy','rb'))
Y_train = np.load(open('Y_train.npy','rb'))
X_test = np.load(open('X_test.npy','rb'))

rs = ShuffleSplit(n_splits=1,test_size=0.2)

for train_index, test_index in rs.split(X_train):
    Xtrain = X_train[train_index,:]
    Xtest = X_train[test_index,:]
    ytrain = Y_train[train_index]
    ytest = Y_train[test_index]


mypca = PCA(n_components=7)
X = mypca.fit_transform(Xtrain)

myclf = RandomForestClassifier(n_estimators=20,max_depth=12)

myclf.fit(X,ytrain)

scores = cross_val_score(myclf,X,ytrain,cv=10,scoring='f1')

print(float(sum(scores))/10)


ypred = myclf.predict(Xtest)
print (confusion_matrix(ytest,ypred))




######################################################

#data = list(Y_pred) + [NAME_OF_GROUP] + [NAME_OF_MODEL]
#data_json = json.dumps(data)
#headers = {'Content-Type': 'application/json'}
#response = requests.post('http://' + IP_SERVER + '/score', data=data_json, headers=headers)
#from tornado import escape
#scores = escape.json_decode(response.content)
#
#print("Your Test Accuracy score is: " + str(round(scores["accuracy"],3)))
#print("Your Test F1 score is: " + str(round(scores["f1"],3)))
#print("Your Test ROC score is: " + str(round(scores["roc"],3)))
#print("Your Test Precision score is: " + str(round(scores["precision"],3)))
#print("Your Test Recall score is: " + str(round(scores["recall"],3)))

