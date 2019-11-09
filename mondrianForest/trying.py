import numpy as np
from skgarden import MondrianForestRegressor, MondrianForestClassifier
from sklearn.datasets import load_iris
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

mfr = MondrianForestClassifier()
mfr.fit(X_train, y_train)
print(mfr.score(X_test, y_test))
# print(dir(mfr))
y_mean = mfr.predict_proba([X_test[0]])
y_pred = mfr.predict([X_test[0]])

print('\n', y_mean[0], sum(y_mean[0]))
print('\n', y_mean, sum(y_mean))

print(y_test)
print("pred :", y_pred)
