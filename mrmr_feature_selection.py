from skfeature.function.information_theoretical_based.MRMR import mrmr

from load_dataset import load_all_image_feature_data

data, labels, cv_generator = load_all_image_feature_data()

use_validation = True
n_features = 512
for training_idxs, validation_idxs in cv_generator:
    split_idx = validation_idxs if use_validation else training_idxs
    print("Running mRMR for image features on {} examples, selecting {} features.".format(len(split_idx), n_features))
    selected_features, _, _ = mrmr(data[split_idx], labels[split_idx], n_selected_features=n_features)
    print(selected_features)
