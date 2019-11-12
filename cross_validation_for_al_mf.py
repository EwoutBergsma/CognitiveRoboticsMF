import numpy as np

from load_dataset import load_vfh_data
from mondrian_forest_classifier_with_al_strategy import MondrianForestClassifierWithALStrategy

data, labels, cv_generator = load_vfh_data()

ini_amount_depth_data_thres = 300
ini_amount_depth_data_perc = 3000
num_trees_depthdata = 22

scores = []
for training_idxs, validation_idxs in cv_generator:
    mf = MondrianForestClassifierWithALStrategy(n_estimators=10)
    mf.fit_using_al_strategy_thres(data[training_idxs], labels[training_idxs], list(range(51)), 300, 0.5)
    scores.append(mf.score(data[validation_idxs], labels[validation_idxs]))

print(scores)
print(np.mean(scores))
