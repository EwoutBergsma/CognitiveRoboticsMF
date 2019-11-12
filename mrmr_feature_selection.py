from skfeature.function.information_theoretical_based.MRMR import mrmr

from load_dataset import load_all_image_feature_data

data, labels, cv_generator = load_all_image_feature_data()

for _, split_idx in cv_generator:
    selected_features, _, _ = mrmr(data[split_idx], labels[split_idx], n_selected_features=512)
    print(selected_features)
