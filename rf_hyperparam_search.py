from pprint import pprint

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from load_dataset import load_vfh_data

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=100, stop=400, num=4)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 50, num=5)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
grid = {'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap}

pprint(grid)

rf = RandomForestClassifier()

data, labels, cv_generator = load_vfh_data()
rf_random_search = RandomizedSearchCV(estimator=rf, param_distributions=grid, n_iter=100, cv=cv_generator, verbose=2,
                                      random_state=42, n_jobs=20)
# rf_grid_search = GridSearchCV(estimator=rf, param_grid=grid, verbose=2, cv=cv_generator, n_jobs=6)

rf_random_search.fit(data, labels)
print(rf_random_search.cv_results_)
print(rf_random_search.best_params_)
print(rf_random_search.best_score_)
print(rf_random_search.cv_results_['mean_test_score'])