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
#X = mypca.fit_transform(Xsc)
X=Xsc

from sklearn.model_selection import ShuffleSplit

rs = ShuffleSplit(n_splits=1,test_size=0.2)

for train_index, test_index in rs.split(X):
    Xtrain = X[train_index,:]
    Xtest = X[test_index,:]
    ytrain = y[train_index]
    ytest = y[test_index]
    
from sklearn.metrics import confusion_matrix

myclf = GaussianNB()

myclf.fit(Xtrain,ytrain)
ypred = myclf.predict(Xtest)
print confusion_matrix(ytest,ypred)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(myclf,X,y,cv=10)























