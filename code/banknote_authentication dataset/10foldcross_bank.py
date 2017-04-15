#10 fold cross validation for banknoteauthentication dataset

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
dataset = loadtxt('banknote_authentication.csv', delimiter=",")
X = dataset[:,0:4]
Y = dataset[:,4]
model = XGBClassifier()
kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))