import sklearn.datasets
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.linear_model
from sklearn.metrics import confusion_matrix, roc_curve, RocCurveDisplay

import numpy as np
import matplotlib.pyplot as plt
import scipy
import time
import math
from tqdm import tqdm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


def logistic(s):
    return 1.0 / (1.0 + np.exp(-s))

def simoid(s):
	return s * (1 - s)
#print(logistic(s))




mesh = np.meshgrid(np.arange(0, 1.1, 0.01), np.arange(0, 1.1, 0.01))[0]

mmesh = np.stack((mesh.flatten(), mesh.T.flatten()))



#mmesh = np.insert(mmesh, [0], [1], axis=0)
print(mmesh)

x, y = sklearn.datasets.make_classification(class_sep=10, n_samples=2166, n_features=8, n_informative=4, n_classes=2,
                                            n_redundant=4, n_clusters_per_class=3, n_repeated=0, random_state=215366)
print(x)
x = x[:,[0,1]]
x = normalize(x)

x, x_test, y, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=215366)



class Layer:
    def __init__(self):
        self.warstwa = []

    class Linear:
        def __init__(self,neurony, wejscia):
            self.m_w = np.random.randn(wejscia, neurony)
            self.v_b = np.zeros((1, neurony))
            self.v_X = None

        def forward(self,v_x):
            self.v_X = v_x

            return (self.v_X@self.m_w) + self.v_b

        def backward(self,blad,eta):
            self.m_w -= eta * (self.v_X.T @ blad)
            self.v_b -= eta * blad
            return blad @ self.m_w.T

    class Activation:
        def __init__(self):
            self.s = None

        def forward(self,s):
            self.s = s

            return logistic(self.s)

        def backward(self, blad,eta):
            return simoid(logistic(self.s))*blad

    def nowa_warstwa(self,neurony,wejscia):
        self.warstwa.append(self.Linear(neurony,wejscia))
        self.warstwa.append(self.Activation())


    def learn(self,epoki,X,Y):
        for a in tqdm(range(epoki)):
            for i in range(len(X)):
                wyjscie = X[i].reshape(1, -1)
                for j in self.warstwa:
                    wyjscie = j.forward(wyjscie)

                blad = 2 * (wyjscie - Y[i])/Y.size


                for j in reversed(self.warstwa):
                    blad = j.backward(blad, eta=0.1)

    def predict(self, X):
        result = X
        for layer in self.warstwa:
            result = layer.forward(result)
        return np.where(result > 0.5, 1, 0)



test = Layer()
def neurony_warstwy(ilosc_wejsc_start,ilosc_neuronow:list, ilosc_warstw:int):
    for a in range(ilosc_warstw):
        if a == 0:
            test.nowa_warstwa(neurony=ilosc_neuronow[a], wejscia=ilosc_wejsc_start)
            print('1',a,ilosc_wejsc_start)
        elif a == ilosc_warstw-1:
            test.nowa_warstwa(neurony=1, wejscia=ilosc_neuronow[a-1])
            print('2',a)
        else:
            test.nowa_warstwa(neurony=ilosc_neuronow[a], wejscia=ilosc_neuronow[a - 1])
            print('3',a)

neurony_warstwy(2,[100,100,100,100,100], 5)

# test.nowa_warstwa(neurony=100, wejscia=2)
# test.nowa_warstwa(neurony=100, wejscia=100)
# test.nowa_warstwa(neurony=100, wejscia=100)
# test.nowa_warstwa(neurony=100, wejscia=100)
#
# test.nowa_warstwa(neurony=1, wejscia=100)


test.learn( 200, x, y)
#predykcja wartości testowych --------------------
u = test.predict(x_test)
#wyswietlanie ------------------------------------
macierz_pomylek = confusion_matrix(y_test, u)
print(macierz_pomylek)

z = test.predict(mmesh.T)
z = z.reshape(mesh.shape)
plt.contourf(mesh, mesh.T, z, alpha=0.3)
plt.scatter(x_test[:, 0], x_test[:, 1], marker="o", c=y_test, s=25, edgecolor="k")

disp = ConfusionMatrixDisplay(confusion_matrix=macierz_pomylek,display_labels= [0,1])
fig, ax = plt.subplots(figsize=(15, 15))
disp.plot(ax=ax)
plt.show()




#print(test.warstwa[0].m_w)
#print(x,y)







