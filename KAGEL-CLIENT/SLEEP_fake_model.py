# -*- coding: utf-8 -*-
"""
Created on Thu May 18 22:05:39 2017

@author: 106380
"""

import numpy as np
import requests
import json

IP_SERVER = '192.168.43.208:5000'
NAME_OF_GROUP = 'MIGRUPO'
NAME_OF_MODEL = 'MIMODELO'

######################################################

X_train = np.load(open('X_train.npy','rb'))
Y_train = np.load(open('Y_train.npy','rb'))
X_test = np.load(open('X_test.npy','rb'))

from sklearn.neighbors import KNeighborsClassifier

myclf = KNeighborsClassifier(n_neighbors = 10,weights='uniform')
myclf.fit(X_train,Y_train)

Y_pred = myclf.predict(X_test)

######################################################

data = list(Y_pred) + [NAME_OF_GROUP] + [NAME_OF_MODEL]
data_json = json.dumps(data)
headers = {'Content-Type': 'application/json'}
response = requests.post('http://' + IP_SERVER + '/score', data=data_json, headers=headers)
from tornado import escape
scores = escape.json_decode(response.content)

print("Your Test Accuracy score is: " + str(round(scores["accuracy"],3)))
print("Your Test F1 score is: " + str(round(scores["f1"],3)))
print("Your Test ROC score is: " + str(round(scores["roc"],3)))
print("Your Test Precision score is: " + str(round(scores["precision"],3)))
print("Your Test Recall score is: " + str(round(scores["recall"],3)))
