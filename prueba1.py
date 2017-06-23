from sklearn.datasets import load_iris

from sklearn.neighbors import KNeighborsClassifier


print "Hola Mundo"

datosIris = load_iris()

print "El numero de instancias es: " + str(len(datosIris.target))
print "El numero de caracteristicas es: " + str(len(datosIris.data[0,:]))

X = datosIris.data
y = datosIris.target

from sklearn.model_selection import ShuffleSplit

rs = ShuffleSplit(n_splits=10,test_size=.25)

scores = []

for train_index, test_index in rs.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    
    Xtraining = X[train_index,:]
    Xtest = X[test_index,:]
    ytraining = y[train_index]
    ytest = y[test_index]
    
    myclf = KNeighborsClassifier(n_neighbors=3)
    
    myclf.fit(Xtraining,ytraining)
    
    ypred = myclf.predict(Xtest)
    scores.append(float(sum(ypred==ytest))/len(ytest))
    
    