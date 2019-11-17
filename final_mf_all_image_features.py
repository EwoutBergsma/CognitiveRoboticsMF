from final_general_functions import cross_validate_mondrian_forest
from load_dataset import load_all_image_feature_data

# Load_vfh_data return the data, labels(targets) and a generator that can be used to fit with 10-fold cross-validation
data, labels, cv_generator = load_all_image_feature_data(use_mRMR=True)

mf_params = {
    'n_estimators': 20,
}
cross_validate_mondrian_forest(data, labels, cv_generator, **mf_params)
