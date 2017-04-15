# gradient boosting implementation for the airfoilselfnoise dataset with both performance(accuracy) and speed(time of execution) 

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
dataset = loadtxt('banknote_authentication.csv', delimiter=",")
X = dataset[:,0:4]
Y = dataset[:,4]
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size,
random_state=seed)
params = {'n_estimators': 100, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01 }
clf = GradientBoostingClassifier(**params)
start_time=time.time()
clf.fit(X_train, y_train)
accuracy = accuracy_score(y_test, clf.predict(X_test))
print("Accuracy: %.4f" % (accuracy* 100.0))
print("%s" %(time.time() - start_time))