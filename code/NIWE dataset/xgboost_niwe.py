#xgboost implementation for niwe dataset including both performance(accuracy) and speed(time of execution)

import time
from numpy import loadtxt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
dataset = loadtxt('data.csv', delimiter=",")
X = dataset[:,0:3]
Y = dataset[:,3]
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size,
random_state=seed)
model = XGBRegressor()
start_time=time.time()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = mean_squared_error(y_test, predictions)
print("Accuracy: %.4f%%" % (accuracy ))
print("%s" %(time.time() - start_time))