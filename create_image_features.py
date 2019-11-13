from os.path import join

import numpy as np
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.models import Model

from utils import cats

dataset_folder = "./new_dataset"

# Initialize the VGG19 network
base_model = VGG19(weights='imagenet')
# Create a model that will output the features from the first dense layer of the VGG19 network
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

for cat in cats:
    # Load in the images for a category the np array shape will be (n, 244, 244, 3)
    print(f"Reading {cat}_smoothed_images.npy")
    smoothed_images = np.load(join(dataset_folder, f"{cat}_smoothed_images.npy"))

    # Changes the color data to 0 centered CNN friendly numbers
    x = preprocess_input(smoothed_images)
    # Feed the CNN the images this will return a np array with shape (n, 4096)
    fc1_features = model.predict(x)

    # Save the features to a binary .npy file
    print(f"Saving {cat}_features.npy with shape: {fc1_features.shape}")
    np.save(join(dataset_folder, f"{cat}_smoothed_image_features.npy"), fc1_features)
