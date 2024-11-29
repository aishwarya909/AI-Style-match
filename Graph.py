from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

# Import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import numpy as np

# Load features from the pickle file
with open('1embeddings1.pkl', 'rb') as f:
    feature_list = pickle.load(f)

# Save the features as a .npy file
np.save('image_features_inception.npy', feature_list)


# Load features
features = np.load('image_features_inception.npy')

# Reduce features to 2D for visualization
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(features)

# Plot t-SNE result
plt.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.5)
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.title('t-SNE Visualization of Image Features')
plt.show()

normalized_feature_list = np.load('image_features_inception.npy')

# Perform PCA for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(normalized_feature_list)

# Save PCA results
pickle.dump(pca_result, open('1pca_results.pkl', 'wb'))

# Visualize PCA results
plt.figure(figsize=(10, 7))
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
plt.title('PCA of Feature Vectors')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()
plt.savefig('pca_visualization.png')  # Save the PCA plot
plt.show()


