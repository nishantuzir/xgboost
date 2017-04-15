# feature importance of the airfoilselfnoise dataset

from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_tree
from matplotlib import pyplot
dataset = loadtxt('airfoil_self_noise.csv', delimiter=",")
X = dataset[:,0:5]
y = dataset[:,5]
model = XGBClassifier()
model.fit(X, y)
plot_tree(model)
pyplot.show()