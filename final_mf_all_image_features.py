from skgarden import MondrianForestClassifier
from sklearn.model_selection import cross_val_score

from load_dataset import load_all_image_feature_data

# Load_vfh_data return the data, labels(targets) and a generator that can be used to fit with 10-fold cross-validation
data, labels, cv_generator = load_all_image_feature_data()
print(data.shape, labels.shape)

# Create model
model = MondrianForestClassifier(n_estimators=20, max_depth=None, min_samples_split=2, bootstrap=False)
# Get the scores
scores = cross_val_score(model, data, labels, cv=cv_generator)
# Print the scores, this will show 10 accuracies when using 10-fold cross-validation
print(scores)
# Print the averaged scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))