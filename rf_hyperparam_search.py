from pprint import pprint

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from load_dataset import load_vfh_data

# Number of trees in random forest
n_estimators = [300, 400]
# Number of features to consider at every split
max_features = ['auto'] # , 'sqrt'
# Maximum number of levels in tree
max_depth = [40, 50, None]  # int(x) for x in np.linspace(10, 50, num=5)]
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

# Load the data, labels and cv idxs generator
data, labels, cv_generator = load_vfh_data()

# Initialize a classifier
rf = RandomForestClassifier()

# The RandomizedSearchCV will try out n_iter random combinations of the supplied grid, n_jobs specifies the amount of
# workers used, don't run 20 workers on a your laptop since this requires tons of RAM.
rf_random_search = RandomizedSearchCV(estimator=rf, param_distributions=grid, n_iter=100, cv=cv_generator, verbose=2,
                                      n_jobs=20)
# rf_grid_search = GridSearchCV(estimator=rf, param_grid=grid, verbose=2, cv=cv_generator, n_jobs=6)

# Run the random hyperparameter search
rf_random_search.fit(data, labels)

# Print all scores for the different setups tried
print(rf_random_search.cv_results_['mean_test_score'])
# Print the params with the best result
print(rf_random_search.best_params_)
# Print the best result
print(rf_random_search.best_score_)

