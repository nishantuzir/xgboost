# feature importance of the banknoteauthentication dataset

from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_tree
from matplotlib import pyplot
dataset = loadtxt('banknote_authentication.csv', delimiter=",")
X = dataset[:,0:4]
y = dataset[:,4]
model = XGBClassifier()
model.fit(X, y)
plot_tree(model)
pyplot.show()