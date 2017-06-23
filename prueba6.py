from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

import colorsys

def get_N_HexCol(N=5):

    HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in xrange(N)]
    hex_out = []
    for rgb in HSV_tuples:
        rgb = map(lambda x: int(x*255),colorsys.hsv_to_rgb(*rgb))
        hex_out.append("".join(map(lambda x: chr(x).encode('hex'),rgb)))
    return hex_out

datosIris = load_digits()

print "El numero de instancias es: " + str(len(datosIris.target))
print "El numero de caracteristicas es: " + str(len(datosIris.data[0,:]))

Xraw = datosIris.data
y = datosIris.target

from sklearn.preprocessing import MinMaxScaler

myMinMaxScaler = MinMaxScaler()
Xsc = myMinMaxScaler.fit_transform(Xraw)

from sklearn.decomposition import PCA

mypca = PCA(n_components=2)
X = mypca.fit_transform(Xsc)

myclf = RandomForestClassifier(n_estimators=30,max_depth=3)

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

#colors = get_N_HexCol(10)#['b', 'g', 'r']
plt.figure()
plt.scatter(X[:,0],-X[:,1],c = y, marker='o', s=80)
