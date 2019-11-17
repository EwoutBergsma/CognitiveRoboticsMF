import pandas as pd
import pymrmr

from load_dataset import load_all_image_feature_data

data, labels, cv_generator = load_all_image_feature_data()

use_validation = False
n_features = 512
for training_idxs, validation_idxs in cv_generator:
    split_idx = validation_idxs if use_validation else training_idxs
    df = pd.DataFrame(data=data[split_idx], index=labels[split_idx], columns=list(map(str, range(data.shape[1]))))
    print("Running mRMR for image features on {} examples, selecting {} features. Using MID.".format(len(split_idx), n_features))
    print(pymrmr.mRMR(df, 'MID', n_features))
