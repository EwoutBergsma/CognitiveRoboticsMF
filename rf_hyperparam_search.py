from sklearn.model_selection import GridSearchCV
import numpy as np
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier


# Number of trees in random forest
from load_dataset import load_simple_vfh_dataset

n_estimators = [int(x) for x in np.linspace(start=100, stop=400, num=4)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 50, num=4)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
grid = {'n_estimators': [100], #n_estimators,
        'max_features': max_features,
        # 'max_depth': max_depth,
        # 'min_samples_split': min_samples_split,
        # 'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap}

pprint(grid)


rf = RandomForestClassifier()



rf_grid_search = GridSearchCV(estimator=rf, param_grid=grid, verbose=2, cv=1)

x_train, y_train, x_test, y_test = load_simple_vfh_dataset()
rf_grid_search.fit(x_train, y_train)
sorted(rf_grid_search.cv_results_.keys())
