import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time  # Import time module for timing

# Initialize EfficientNetB0 model
model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Function to extract features with timing
def extract_features_with_timing(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)

    # Measure inference time
    start_time = time.time()
    result = model.predict(preprocessed_img).flatten()
    end_time = time.time()
    inference_time = end_time - start_time

    normalized_result = result / norm(result)
    return normalized_result, inference_time

# Define lists to store features and inference times
feature_list = []
inference_times = []

# Extract and normalize features, also track inference times
filenames = [os.path.join('1ima', file) for file in os.listdir('1ima')]

for file in tqdm(filenames):
    features, inference_time = extract_features_with_timing(file, model)
    feature_list.append(features)
    inference_times.append(inference_time)

# Normalize the extracted feature list
normalized_feature_list = normalize(feature_list, norm='l2')

# Save the features, filenames, and inference times
pickle.dump(normalized_feature_list, open('1embeddings1.pkl', 'wb'))
pickle.dump(filenames, open('1filenames1.pkl', 'wb'))
pickle.dump(inference_times, open('1inference_times.pkl', 'wb'))  # Save inference times


# Print average inference time
average_inference_time = np.mean(inference_times)
print(f"Average inference time per image: {average_inference_time:.4f} seconds")
