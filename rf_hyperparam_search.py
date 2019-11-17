from pprint import pprint
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from load_dataset import load_vfh_data, load_good5_data, load_good15_data

# Number of trees in random forest
n_estimators = [300, 400]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [10, 20, 30]  # int(x) for x in np.linspace(10, 50, num=5)]
# Minimum number of samples required to split a node
min_samples_split = [2] # , 5
# Minimum number of samples required at each leaf node
min_samples_leaf = [1] # , 2
# Method of selecting samples for training each tree
bootstrap = [False]

# Create the random grid
grid = {'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap}
#grid = {'n_estimators': [400, 500], 'max_depth': [None]}
pprint(grid)

# Load the data, labels and cv idxs generator
try:
    shape_descriptor = int(sys.argv[1])
except IndexError:
    shape_descriptor = 2

try:
    amount_of_jobs = int(sys.argv[2])
except IndexError:
    amount_of_jobs = 10
print("using shape descriptor {}, using {} cpus".format(shape_descriptor, amount_of_jobs))

if shape_descriptor == 0:
    data, labels, cv_generator = load_vfh_data()
elif shape_descriptor == 1:
    data, labels, cv_generator = load_good5_data()
else:
    data, labels, cv_generator = load_good15_data()


# Initialize a classifier
rf = RandomForestClassifier()

# The RandomizedSearchCV will try out n_iter random combinations of the supplied grid, n_jobs specifies the amount of
# workers used, don't run 20 workers on a your laptop since this requires tons of RAM.
#rf_random_search = RandomizedSearchCV(estimator=rf, param_distributions=grid, n_iter=100, cv=cv_generator, verbose=2,
#                                      n_jobs=amount_of_jobs)
rf_grid_search = GridSearchCV(estimator=rf, param_grid=grid, verbose=2, cv=cv_generator, n_jobs=amount_of_jobs)

# Run the random hyperparameter search
rf_grid_search.fit(data, labels)

# Print all scores for the different setups tried
print(rf_grid_search.cv_results_['mean_test_score'])
# Print the params with the best result
print(rf_grid_search.best_params_)
# Print the best result
print(rf_grid_search.best_score_)
