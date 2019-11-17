import mifs

from load_dataset import load_all_image_feature_data

data, labels, cv_generator = load_all_image_feature_data()

use_validation = False
n_features = 500
for training_idxs, validation_idxs in cv_generator:
    split_idx = validation_idxs if use_validation else training_idxs
    print("Running mRMR for image features on {} examples, selecting {} features. Using MIFS.".format(len(split_idx), n_features))
    X = data[split_idx]
    y = labels[split_idx]
    feat_selector = mifs.MutualInformationFeatureSelector(method='MRMR', n_features=500, n_jobs=10, verbose=2)
    feat_selector.fit(X, y)
    print(feat_selector.support_)
    print(feat_selector.ranking_)
