import numpy as np

from load_dataset import load_vfh_data
from mondrianForest.cleaned_up_active_learning import MondrianForestClassifierWithALStrategy

data, labels, cv_generator = load_vfh_data()

ini_amount_depth_data_thres = 300
ini_amount_depth_data_perc = 3000
num_trees_depthdata = 22

scores = []
for training_idxs, validation_idxs in cv_generator:
    mf = MondrianForestClassifierWithALStrategy(n_estimators=22)
    mf.fit_using_al_strategy_thres(data[training_idxs], labels[training_idxs], list(range(51)), 300, 0.5)
    scores.append(mf.score(data[validation_idxs], labels[validation_idxs]))

print(scores)
print(np.mean(scores))
