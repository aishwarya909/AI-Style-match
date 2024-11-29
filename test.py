import pickle
import tensorflow as tf
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2
import matplotlib.pyplot as plt
import time

# Load features, filenames, and PCA results
feature_list = np.array(pickle.load(open('1embeddings1.pkl', 'rb')))
filenames = pickle.load(open('1filenames1.pkl', 'rb'))
pca_result = np.array(pickle.load(open('1pca_results.pkl', 'rb')))

# Initialize EfficientNetB0 model for prediction
model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Extract features of a sample image for recommendation
img = image.load_img('sample/S1.png', target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

# K-Nearest Neighbors for Similarity Search
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)

distances, indices = neighbors.kneighbors([normalized_result])
print("Similar items indices:", indices)

# Similarity Quality metric (average distance of top matches)
similarity_quality = np.mean(distances[0][1:6])  # Ignore the first distance as it's the image itself
print(f"Similarity Quality (average distance): {similarity_quality:.4f}")



# Display recommended images
for file_index in indices[0][1:6]:  # Skip the first result as it is the input image itself
    temp_img = cv2.imread(filenames[file_index])
    cv2.imshow('Recommended Image', cv2.resize(temp_img, (512, 512)))
    cv2.waitKey(0)
cv2.destroyAllWindows()

# Function to calculate average distance
def calculate_average_distance(neighbors, input_feature, k):
    distances, indices = neighbors.kneighbors([input_feature])
    avg_distance = np.mean(distances[0])
    return avg_distance

# Calculate average distance for EfficientNet Model
avg_distance = calculate_average_distance(neighbors, normalized_result, 5)
print("ResNet Model - Average Distance:", avg_distance)

import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
import tensorflow as tf
from numpy.linalg import norm
from sklearn.metrics.pairwise import euclidean_distances

# Load precomputed features and filenames
feature_list = np.array(pickle.load(open('1embeddings1.pkl', 'rb')))
filenames = pickle.load(open('1filenames1.pkl', 'rb'))

# Initialize the EfficientNetB0 model
model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def extract_features(img_path):
    """Extract normalized feature vector for a single image."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def get_top_k_similar_images(query_image_path, feature_list, filenames, k=5):
    """Retrieve top-K similar images and their feature vectors."""
    query_features = extract_features(query_image_path)
    neighbors = NearestNeighbors(n_neighbors=k + 1, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([query_features])
    top_k_indices = indices[0][1:]
    top_k_features = feature_list[top_k_indices]
    top_k_filenames = [filenames[i] for i in top_k_indices]
    return top_k_features, top_k_filenames

def calculate_diversity(feature_vectors):
    """Calculate the diversity of the retrieved images based on pairwise distances."""
    pairwise_distances = euclidean_distances(feature_vectors)
    diversity_score = np.sum(np.triu(pairwise_distances, k=1))
    num_pairs = len(feature_vectors) * (len(feature_vectors) - 1) / 2
    average_diversity = diversity_score / num_pairs
    return average_diversity

# Example usage
query_image_path = 'sample/S1.png'  # Path to the query image
k = 5  # Number of similar images to retrieve

# Get top-K similar images and their feature vectors
top_k_features, top_k_filenames = get_top_k_similar_images(query_image_path, feature_list, filenames, k)

# Calculate the diversity of the retrieved images
diversity_score = calculate_diversity(top_k_features)
print(f"Diversity of Retrieved Images: {diversity_score:.4f}")

