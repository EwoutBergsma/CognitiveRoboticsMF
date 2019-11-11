from skgarden import MondrianForestClassifier
from sklearn.model_selection import cross_val_score


def cross_validate_mondrian_forest(data, labels, cv_generator, **mf_params):
    print("Data and labels shape: ", data.shape, labels.shape)
    # Create model
    model = MondrianForestClassifier(**mf_params)
    # Get the scores
    scores = cross_val_score(model, data, labels, cv=cv_generator, verbose=2)
    # Print the scores, this will show 10 accuracies when using 10-fold cross-validation
    print(scores)
    # Print the averaged scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))