from sklearn.datasets import load_digits
from sklearn.naive_bayes import GaussianNB

datosIris = load_digits()

print "El numero de instancias es: " + str(len(datosIris.target))
print "El numero de caracteristicas es: " + str(len(datosIris.data[0,:]))

Xraw = datosIris.data
y = datosIris.target

from sklearn.preprocessing import MinMaxScaler

myMinMaxScaler = MinMaxScaler()
Xsc = myMinMaxScaler.fit_transform(Xraw)

from sklearn.decomposition import PCA

mypca = PCA(n_components=10)
X = mypca.fit_transform(Xsc)

from sklearn.model_selection import cross_val_score

myclf = GaussianNB()
scores = cross_val_score(myclf,X,y,cv=10)
'''

#rejilla_parametros = {'n_neighbors': range(3,15), 'weights': ['uniform','distance'], 'p': [2,3]}

#from sklearn.model_selection import GridSearchCV


#mygridsearchcv = GridSearchCV(myclf,rejilla_parametros,cv=10)
#mygridsearchcv.fit(X,y)

#myclf = mygridsearchcv.best_estimator_
myclf.fit(X,y)

import numpy as np

minimos = np.min(X,axis=0)
maximos = np.max(X,axis=0)

NPOINTS = 50

vector_x0 = np.linspace(minimos[0],maximos[0],NPOINTS)
vector_x1 = np.linspace(minimos[1],maximos[1],NPOINTS)

xx0,xx1 = np.meshgrid(vector_x0,vector_x1)

predictionsRejilla = np.zeros((NPOINTS,NPOINTS))

for i in range(NPOINTS):
    for j in range(NPOINTS):
        predictionsRejilla[i,j]=myclf.predict([xx0[i,j],xx1[i,j]])
        
from matplotlib import pyplot as plt

plt.figure()
plt.matshow(predictionsRejilla)

markers = ['o','s','x']
colors = ['b', 'g', 'r']

plt.figure()
for k in range(len(X)):
    plt.scatter(X[k,0],X[k,1],c = colors[y[k]], marker=markers[y[k]])

plt.show()'''