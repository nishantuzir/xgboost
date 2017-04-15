# feature importance for the pimaindiansdiabetes dataset

from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_tree
from matplotlib import pyplot
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
X = dataset[:,0:8]
y = dataset[:,8]
model = XGBClassifier()
model.fit(X, y)
plot_tree(model)
pyplot.show()