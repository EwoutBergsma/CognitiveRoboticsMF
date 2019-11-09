from sklearn import svm
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

from load_dataset import load_old_dataset, load_simple_vfh_dataset

# Load the data
x_train, y_train, x_test, y_test = load_simple_vfh_dataset()

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# # Create default Linear SVM
# model = svm.SVC(kernel='linear')
model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print(score)

# # Define cross validation settings
# cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
# # Cross validate
# scores = cross_val_score(clf, vfhs_reps, labels, cv=cv)
# # Print results
# print(scores)
# print("Accuracy: %0.2f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
