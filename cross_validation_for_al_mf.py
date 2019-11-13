import numpy as np

from load_dataset import load_vfh_data
from mondrian_forest_classifier_with_al_strategy import MondrianForestClassifierWithALStrategy
import sys

ini_amount_depth_data_thres = 300
ini_amount_depth_data_perc = 3000
num_trees_depth_data = 22

data, labels, cv_generator = load_vfh_data()

threshold = float(sys.argv[1])
scores = []
samples_used = []

print("\nPerforming cross-validation for vfh data with AL using a threshold of {}".format(threshold))

for training_idxs, validation_idxs in cv_generator:
    mf = MondrianForestClassifierWithALStrategy(n_estimators=22)
    samples_used.append(mf.fit_using_al_strategy_thres(data[training_idxs], labels[training_idxs], np.array(range(51)),
                                                       300, threshold))
    scores.append(mf.score(np.array(data[validation_idxs, :]), np.array(labels[validation_idxs])))
    print(scores[-1], samples_used[-1])
    del mf

print("Samples used: ", np.mean(scores), "+-", np.std(scores))
print(samples_used)

print("Accuracies: ", np.mean(samples_used), "+-", np.std(samples_used))
print(scores)
