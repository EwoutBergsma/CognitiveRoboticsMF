import numpy as np
import random
from numpy import array
from skgarden import MondrianForestRegressor, MondrianForestClassifier
from sklearn.datasets import load_iris
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from load_dataset import load_simple_vfh_dataset
"""
    X_train/test : num of dataset * features 
                    (34772, 308) / (7104, 308) 
    y_train/test : labels of dataset 
                    (34772,) / (7104,)
"""
X_train, y_train, X_test, y_test = load_simple_vfh_dataset("/home/sohyung/Documents/19-20/1A/CognitiveRobotics/final_project/CognitiveRoboticsMF/new_dataset_only_vfh")
# data = np.insert(X_train, y_train, axis=1)
# print(X_train)
# print(y_train)
# y = y_train.reshape((-1, 1))
# print(y)
# data = np.concatenate((X_train, y), axis=1)
# print(data)
# print(data.shape)
#
# data = np.concatenate((X_train, y_train.reshape((-1, 1))), axis=1)
# print("data :", data)
# row_i = np.random.choice(data.shape[0], 30)
# print(row_i)
# print(data[row_i, :])
# print(data[row_i, :].shape)


def random_sampling(X_train, y_train, num_of_dataset):
    data = np.concatenate((X_train, y_train.reshape((-1, 1))), axis=1)
    rand_selected_rows = np.random.choice(data.shape[0], num_of_dataset)
    rand_samples = data[rand_selected_rows, :]
    return rand_samples[:, :-1], rand_samples[:, -1]

sampled_X_train, sampled_y_train = random_sampling(X_train, y_train, 300)
print("test datasets :", sampled_X_train)
print(sampled_X_train.shape)
print("labels :", sampled_y_train)
print(sampled_y_train.shape)
###############
# try 10 trees
mfr = MondrianForestClassifier(n_estimators=10)
mfr.fit(X_train, y_train)
print(mfr.score(X_test, y_test))

# print(dir(mfr))
y_mean = mfr.predict_proba([X_test[0]])
y_pred = mfr.predict([X_test[0]])

y_pred_proba = mfr.predict_proba([X_test])
for test in X_test:
    y_pred = mfr.predict([test])
    print("pred:", y_pred)

print("pred :", y_pred)
print("pred_proba : ", y_pred_proba)
# print('\n', y_mean[0], sum(y_mean[0]))
# print('\n', y_mean, sum(y_mean))


####################################################################################################

######trying part ######
print("start")
active_learning_thres(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, num_trees=num_trees_depthdata, threshold=0.5 )
sampled_X_train, sampled_y_train, selected_training_data = random_sampling(X_train=X_train, y_train=y_train, num_of_dataset=30000)
mfc = MondrianForestClassifier(n_estimators=10)
mfc.partial_fit(sampled_X_train, sampled_y_train)
rows_remaining_train_data = [i for i in range(0, X_train.shape[0]) if i not in selected_training_data]
remaining_data = X_train[rows_remaining_train_data, :]
pre_my = mfc.predict_proba(X_train[rows_remaining_train_data, :])
predict_proba = mfc.predict_proba([remaining_data[0]])
desend = sorted(pre_my[0], reverse=True)
sr = sorted(pre_my[0])
hehe = sorted(predict_proba[0], reverse=True)
print(pre_my[0])
print("works? desending :", desend)
print("works? :", sr)
# print(pre_my[0])
print(hehe)
# it's the same :P


# sampled_X_train, sampled_y_train, selected_training_data = random_sampling(X_train=X_train, y_train=y_train, num_of_dataset=30000)
# mfc = MondrianForestClassifier(n_estimators=10)
# mfc.partial_fit(sampled_X_train, sampled_y_train)
# # mfc.fit(sampled_X_train, sampled_y_train)
# rows_remaining_train_data = [i for i in range(0, X_train.shape[0]) if i not in selected_training_data]
# #The class probabilities of the input samples. The ord  File "<input>", line 2, in <module>
# TypeError: list indices must be integers or slices, not list
# >>> for rows in hehe:
# ...     print(rows)
# ...
# [1, 2, 3]
# [4, 5, 6]
# [7, 8, 9]
# [10, 11, 12]
# >>> sorted(a)
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# >>> sorted(a, reverse=True)
# [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
# >>> haha = [[2.1, 3.2, 4.5],[3.002, 0.113, 3.21],[0.62,0.3321,3.1]]
# >>> for rows in haha:
# ...     sorted_row = sorted(rows,revers=True)
# ...     print(sorted_row)
# ...
# Traceback (most recent call last):
#   File "<input>", line 2, in <module>
# TypeError: 'revers' is an invalid keyword argument for this function
# >>> for rows in haha:
# ...     sorted_row = sorted(rows,reverse=True)
# ...     print(sorted_row)
# ...
# [4.5, 3.2, 2.1]
# [3.21, 3.002, 0.113]
# [3.1, 0.62, 0.3321]
# >>> arrray(haha)
# Traceback (most recent call last):
#   File "<input>", line 1, in <module>
# NameError: name 'arrray' is not defineder of the
# # classes corresponds to that in the attribute `classes_`.
# predict_proba = mfc.predict_proba(X_train[rows_remaining_train_data, :])


#for probabilities in predict_proba:
#    print("prob", probabilities)
    # confidence = calculate_confidence(probabilities)
    # p  = probabilities.sort(reverse=True)
    # print("confidence: ", confidence)

# y_pred = mfr.predict(X_train)
# y_pred_proba = mfr.predict_proba(sampled_X_train)
# print(mfr.score(X_test, y_test))
# print("pred :", y_pred)
# print("pred_proba : ", y_pred_proba)


############# non-using functions ##################

# def active_learning(ini_amount_depth_data_thres, X_train, y_train, X_test, y_test, num_trees=num_trees_depthdata,
#                     threshold, confidence):
#     # initial set up of data
#     sampled_X_train, sampled_y_train = random_sampling(X_train, y_train, ini_amount_depth_data_thres)
#     # the rirst adaptation training
#     y_pred, accuracy = mondiran_forest(num_trees, sampled_X_train, sampled_y_train, X_test, y_test)
#
#     # two different strategy
#     # threshold
#     # p
#     # active_learning_thres()
#     # active_learning_prec()

# def random_sampling(X_train, y_train, num_of_dataset):
#     # data = np.concatenate((X_train, y_train.reshape((-1, 1))), axis=1)
#     # rand_selected_rows = np.random.choice(data.shape[0], num_of_dataset)
#     rand_selected_rows = np.random.choice(X_train.shape[0], num_of_dataset)
#     # rand_samples = X_train[rand_selected_rows, :]
#     # return rand_samples[:, :-1], rand_samples[:, -1]
#     return X_train[rand_selected_rows, :], y_train[rand_selected_rows]
