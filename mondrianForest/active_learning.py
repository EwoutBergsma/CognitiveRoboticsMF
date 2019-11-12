import numpy as np
import random
from skgarden import MondrianForestRegressor, MondrianForestClassifier
from sklearn.datasets import load_iris
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from load_dataset import load_simple_vfh_dataset
from sklearn.model_selection import cross_val_score


"""
    >> description of our dataset
    
    51 labels
    X_train/test : num of dataset * features(308) 
                    (34772, 308) / (7104, 308) 
    y_train/test : contents == labels of dataset
                    (34772,) / (7104,)
"""
X_train, y_train, X_test, y_test = load_simple_vfh_dataset("/home/sohyung/Documents/19-20/1A/CognitiveRobotics/final_project/CognitiveRoboticsMF/new_dataset_only_vfh")

# parameters : now I'm just following from the reference
ini_amount_depth_data_thres = 300
ini_amount_depth_data_perc = 3000
num_trees_depthdata = 22


def calculate_confidence(probabilities):
    """
    :param probabilities:
    :return confidence:
    """
    sorted_proba = sorted(probabilities, reverse=True)
    return 1 - sorted_proba[1]/sorted_proba[0]


def random_sampling(X_train, y_train, num_of_dataset):
    rand_selected_rows = np.random.choice(X_train.shape[0], num_of_dataset, replace=False)
    return X_train[rand_selected_rows, :], y_train[rand_selected_rows], rand_selected_rows



def mondiran_forest(num_trees, X_train, y_train, X_test, y_test):
    mfc = MondrianForestClassifier(n_estimators=num_trees)
    mfc.partial_fit(X_train, y_train)
    # y_mean = mfr.predict_proba([X_test[0]])
    # y_pred = mfr.predict(X_test)
    return y_pred, mfc.score(X_test, y_test)


def active_learning_thres( X_train, y_train, X_test, y_test, ini_amount_depth_data_thres = ini_amount_depth_data_thres, num_trees=num_trees_depthdata, threshold=0.5):
    # first, get the initial data set
    sampled_X_train, sampled_y_train, init_training_data = random_sampling(X_train, y_train, ini_amount_depth_data_thres)

    # first round of adaptation training
    mfc = MondrianForestClassifier(n_estimators=num_trees)
    mfc.partial_fit(sampled_X_train, sampled_y_train)

    # Now get another samples from remaining training data
    rows_remaining_train_data = [i for i in range(0, X_train.shape[0]) if i not in init_training_data]

    # all_instances_proba = mfc.predict_proba(X_train[rows_remaining_train_data, :])
    # for probabilities in all_instances_proba:
    #     if calculate_confidence(probabilities) < threshold:

    remaining_X_data = X_train[rows_remaining_train_data, :]
    remaining_y_data = y_train[rows_remaining_train_data]
    selected_data = np.concatenate((remaining_X_data, remaining_y_data.reshape((-1, 1))), axis=1)
    # TODO : is there any better way to do this? :(

    selected_X_data = []
    selected_y_data = []
    for instance in selected_data:
        probabilities = mfc.predict_proba([instance[:-1]])
        if calculate_confidence(probabilities[0]) < threshold:
            selected_X_data.append(instance[:-1])
            selected_y_data.append(instance[-1])
    mfc.partial_fit(selected_X_data, selected_y_data)
    y_pred = mfc.predict(X_test)
    return mfc.score(X_test, y_test), len(selected_y_data)

def active_learning_prec():
    pass

accuracy = active_learning_thres(ini_amount_depth_data_thres = 3000, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
print(accuracy)

#
# def mondiran_forest(num_trees, X_train, y_train, X_test, y_test):
#     mfc = MondrianForestClassifier(n_estimators=num_trees)
#     mfc.partial_fit(X_train, y_train)
#     # y_mean = mfr.predict_proba([X_test[0]])
#     # y_pred = mfr.predict(X_test)
#     return y_pred, mfc.score(X_test, y_test)
