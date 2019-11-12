import numpy as np

from load_dataset import load_vfh_data
from mondrian_forest_classifier_with_al_strategy import MondrianForestClassifierWithALStrategy


ini_amount_depth_data_thres = 300
ini_amount_depth_data_perc = 3000
num_trees_depth_data = 22

data, labels, cv_generator = load_vfh_data()

scores = []
for training_idxs, validation_idxs in cv_generator:
    mf = MondrianForestClassifierWithALStrategy(n_estimators=22)
    mf.fit_using_al_strategy_thres(data[training_idxs], labels[training_idxs], np.array(range(51)), 300, 0.5)
    scores.append(mf.score(np.array(data[validation_idxs, :]), np.array(labels[validation_idxs])))
    print(scores[-1])

print('\n\n', scores)
print(np.mean(scores))
