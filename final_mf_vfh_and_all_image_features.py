from pprint import pprint

from skgarden import MondrianForestClassifier
from sklearn.model_selection import GridSearchCV

from load_dataset import load_vfh_and_all_image_feature_data

# Number of trees in random forest
n_estimators = [20]
# Maximum number of levels in tree
max_depth = [None]  # int(x) for x in np.linspace(10, 50, num=5)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
grid = {'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'bootstrap': bootstrap}

pprint(grid)

# Load the data, labels and cv idxs generator
data, labels, cv_generator = load_vfh_and_all_image_feature_data()

# Initialize a classifier
mondrian_forest = MondrianForestClassifier()

# The RandomizedSearchCV will try out n_iter random combinations of the supplied grid, n_jobs specifies the amount of
# workers used, don't run 20 workers on a your laptop since this requires tons of RAM.
# rf_random_search = RandomizedSearchCV(estimator=mondrian_forest, param_distributions=grid, n_iter=100,
#                                       cv=cv_generator, verbose=2, n_jobs=20)

rf_grid_search = GridSearchCV(estimator=mondrian_forest, param_grid=grid, verbose=2, cv=cv_generator, n_jobs=10)

# Run the random hyperparameter search
rf_grid_search.fit(data, labels)

# Print the params with the best result
hashtags = "\n" + "#"*50 + "\n"
print(
        "{}\nBest score: {}\nParameters used for that score: {}\nOther scores: {}\n{}".format(
                hashtags,
                rf_grid_search.best_score_,
                rf_grid_search.best_params_,
                rf_grid_search.cv_results_['mean_test_score'],
                hashtags
        )
)
