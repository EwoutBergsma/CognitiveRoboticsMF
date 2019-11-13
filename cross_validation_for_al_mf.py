import sys
import time

import numpy as np

from load_dataset import load_vfh_data
from mondrian_forest_classifier_with_al_strategy import MondrianForestClassifierWithALStrategy

ini_amount_depth_data_thres = 300
ini_amount_depth_data_perc = 3000
num_trees_depth_data = 22

data, labels, cv_generator = load_vfh_data()

threshold = float(sys.argv[1])
scores = []
samples_used = []

start_time = time.time()
for training_idxs, validation_idxs in cv_generator:
    mf = MondrianForestClassifierWithALStrategy(n_estimators=22)
    samples_used.append(mf.fit_using_al_strategy_thres(data[training_idxs], labels[training_idxs], np.array(range(51)),
                                                       300, threshold))
    scores.append(mf.score(np.array(data[validation_idxs, :]), np.array(labels[validation_idxs])))
    del mf

print("--------")
print("\nPerforming cross-validation for vfh data with AL using a threshold of {}".format(threshold))
print("The cross validation took {} seconds".format(time.time() - start_time))
print("--------\nSamples used: {:.2f} +- {.2f}".format(np.mean(samples_used), np.std(samples_used)))
print(samples_used)
print("--------\nAccuracies: {:.2f} +- {.2f}".format(np.mean(scores) * 100, np.std(scores) * 100))
print(scores)
print("--------")
