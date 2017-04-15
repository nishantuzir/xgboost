# feature importance in NIWE dataset

from numpy import loadtxt
from xgboost import XGBRegressor
from matplotlib import pyplot
dataset = loadtxt('data.csv', delimiter=",")
X = dataset[:,0:3]
y = dataset[:,3]
model = XGBRegressor()
model.fit(X, y)
print(model.feature_importances_)
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
pyplot.show()