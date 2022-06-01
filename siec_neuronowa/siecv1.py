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

# y = np.array([y]).T
#
# #x = np.insert(x, [0], [1], axis=1)
#
# #x_test = np.insert(x_test, [0], [1], axis=1)
#
#
# # w1 = np.ones([2, 2])
# # w2 = np.ones([2, 1])
# #
# # y1 = y[1]
# # x1 = x[1]
#
# #print(w1,'\n','\n',w2,'\n','\n',y1,'\n','\n',x1)


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


    # def predykcja(self,x_test):
    #     iksy = x_test
    #     for i in self.warstwa:
    #         iksy = i.forward(iksy)





test = Layer()
test.nowa_warstwa(neurony=100, wejscia=2)
test.nowa_warstwa(neurony=100, wejscia=100)
test.nowa_warstwa(neurony=100, wejscia=100)
test.nowa_warstwa(neurony=100, wejscia=100)

test.nowa_warstwa(neurony=1, wejscia=100)


test.learn( 200, x, y)
#predykcja wartości testowych --------------------
u = test.predict(x_test)
#wyswietlanie ------------------------------------
z = test.predict(mmesh.T)
z = z.reshape(mesh.shape)
plt.contourf(mesh, mesh.T, z, alpha=0.3)
plt.scatter(x_test[:, 0], x_test[:, 1], marker="o", c=y_test, s=25, edgecolor="k")
plt.show()


print(u.T[0] == y_test)

#print(test.warstwa[0].m_w)
#print(x,y)







