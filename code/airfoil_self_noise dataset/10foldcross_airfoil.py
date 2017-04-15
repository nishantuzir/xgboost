#10 fold cross validation for airfoilselfnoise dataset

from numpy import loadtxt
from xgboost import XGBRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
dataset = loadtxt('airfoil_self_noise.csv', delimiter=",")
X = dataset[:,0:5]
Y = dataset[:,5]
model = XGBRegressor()
kfold = StratifiedKFold(n_splits=4, random_state=7)
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))