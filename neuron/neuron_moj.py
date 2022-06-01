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


def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


def logistic(s):
    return 1 / (1 + (math.e) ** (-s))


mesh = np.meshgrid(np.arange(0, 1.1, 0.01), np.arange(0, 1.1, 0.01))[0]

mmesh = np.stack((mesh.flatten(), mesh.T.flatten()))

print(mmesh)

mmesh = np.insert(mmesh, [0], [1], axis=0)

# x, y = sklearn.datasets.make_classification(class_sep=1, n_samples=2166, n_features=2, n_informative=2, n_classes=2,
#                                             n_redundant=0, n_clusters_per_class=1, n_repeated=0, random_state=215366)

x, y = sklearn.datasets.make_moons(random_state=215366, n_samples=2166, noise = 0.1)
x = normalize(x)

x, x_test, y, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=215366)

# y[y==0] = -1

y = np.array([y]).T

x = np.insert(x, [0], [1], axis=1)

x_test = np.insert(x_test, [0], [1], axis=1)

teta = np.ones([x_test.shape[1], 1])

start = time.perf_counter()
#print('sdasd', (np.array([x[0]]).reshape(1, -1)).shape)
#print( ((np.array([(-0.07 * (f_s - y[0]) * (1 - f_s) * f_s)]).reshape(1, -1) @ (np.array([x[0]]).reshape(1, -1))).T).shape)

for i in range(10):
    for a in range(len(x)):
        s = x[a] @ teta
        print(s)
        f_s = logistic(s)
        new_teta = (
            (np.array([(-0.07 * (f_s - y[a]) * (1 - f_s) * f_s)]).reshape(1, -1) @ (np.array([x[a]]).reshape(1, -1))).T)
        teta += new_teta
print(teta)

teta = teta.flatten()

wyjscie = logistic(teta @ mmesh)
stop = time.perf_counter()
print('czas wykonywania = ', (stop - start))

wyjscie = np.array([1 if x >= 0.5 else 0 for x in wyjscie])
wyjscie = wyjscie.reshape(mesh.shape)


#przewidziane_klasy= logistic(teta @ x_test.T)
przewidziane_klasy = np.array([1 if x >= 0 else 0 for x in teta @ x_test.T])

print(wyjscie)

macierz_pomylek = confusion_matrix(y_test, przewidziane_klasy)
print(macierz_pomylek)

# #---------------wykreslanie---------------

z = wyjscie
plt.figure(1)
plt.contourf(mesh, mesh.T, z, alpha=0.3)
plt.scatter(x_test[:, 1], x_test[:, 2], marker="o", c=y_test, s=25, edgecolor="k")
#plt.show()



def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


def logistic(s):
    return 1 / (1 + (math.e) ** (-s))


mesh = np.meshgrid(np.arange(0, 1.1, 0.01), np.arange(0, 1.1, 0.01))[0]

mmesh = np.stack((mesh.flatten(), mesh.T.flatten()))

print(mmesh)

mmesh = np.insert(mmesh, [0], [1], axis=0)

x, y = sklearn.datasets.make_classification(class_sep=1, n_samples=2166, n_features=2, n_informative=2, n_classes=2,
                                            n_redundant=0, n_clusters_per_class=1, n_repeated=0, random_state=215366)


x = normalize(x)

x, x_test, y, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=215366)

# y[y==0] = -1


y = np.array([y]).T

x = np.insert(x, [0], [1], axis=1)

x_test = np.insert(x_test, [0], [1], axis=1)

teta = np.ones([x_test.shape[1], 1])

start = time.perf_counter()
#print('sdasd', (np.array([x[0]]).reshape(1, -1)).shape)
#print( ((np.array([(-0.07 * (f_s - y[0]) * (1 - f_s) * f_s)]).reshape(1, -1) @ (np.array([x[0]]).reshape(1, -1))).T).shape)

for i in range(10):
    for a in range(len(x)):
        s = x[a] @ teta
        f_s = logistic(s)
        new_teta = (
            (np.array([(-0.07 * (f_s - y[a]) * (1 - f_s) * f_s)]).reshape(1, -1) @ (np.array([x[a]]).reshape(1, -1))).T)
        teta += new_teta
print(teta)

teta = teta.flatten()

wyjscie = logistic(teta @ mmesh)
stop = time.perf_counter()
print('czas wykonywania = ', (stop - start))

wyjscie = np.array([1 if x >= 0.5 else 0 for x in wyjscie])
wyjscie = wyjscie.reshape(mesh.shape)


#przewidziane_klasy= logistic(teta @ x_test.T)
przewidziane_klasy = np.array([1 if x >= 0 else 0 for x in teta @ x_test.T])



macierz_pomylek = confusion_matrix(y_test, przewidziane_klasy)
print(macierz_pomylek)

# #---------------wykreslanie---------------

z = wyjscie

plt.figure(2)
plt.contourf(mesh, mesh.T, z, alpha=0.3)
plt.scatter(x_test[:, 1], x_test[:, 2], marker="o", c=y_test, s=25, edgecolor="k")
plt.show()
