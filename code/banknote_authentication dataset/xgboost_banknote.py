#xgboost implementation for the banknoteauthentication dataset involving both performance(accuracy) and speed(time of execution)

import time
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
dataset = loadtxt('banknote_authentication.csv', delimiter=",")
X = dataset[:,0:4]
Y = dataset[:,4]
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size,
random_state=seed)
model = XGBClassifier()
start_time=time.time()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("%s" %(time.time() - start_time))