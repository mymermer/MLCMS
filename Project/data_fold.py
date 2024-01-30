import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from datafold.pcfold import PCManifold
import random

def pca_visualization(model, subset_size=None, n_components=2):
    # Get all words from the vocabulary
    all_words = list(model.key_to_index.keys())

    # If a subset size is specified, randomly sample the words
    if subset_size is not None:
        all_words = random.sample(all_words, subset_size)

    # Create a numpy array of word vectors
    word_vectors = np.array([model[word] for word in all_words])

    # Create a PCManifold object and optimize parameters
    pcm = PCManifold(word_vectors)
    pcm.optimize_parameters(result_scaling=2)

    # Apply PCA on the data
    pca = PCA(n_components=n_components)
    transformed_vectors = pca.fit_transform(word_vectors)

    # Plot the result
    plt.figure(figsize=(10, 8))
    plt.scatter(transformed_vectors[:, 0], transformed_vectors[:, 1], marker='.', s=1)
    plt.title("PCA Visualization")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()
