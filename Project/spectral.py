import numpy as np
from sklearn.manifold import SpectralEmbedding
import matplotlib.pyplot as plt
import random
import pyamg

def spectral_embedding_visualization(model, subset_size=None, n_components=2, eigen_solver='amg'):
    # Get all words from the vocabulary
    all_words = list(model.key_to_index.keys())

    # If a subset size is specified, randomly sample the words
    if subset_size is not None:
        all_words = random.sample(all_words, subset_size)

    # Create a numpy array of word vectors
    word_vectors = np.array([model[word] for word in all_words])

    # Create a Spectral Embedding object
    embedding = SpectralEmbedding(n_components=n_components, eigen_solver=eigen_solver)

    # Fit and transform the data
    transformed_vectors = embedding.fit_transform(word_vectors)

    # Plot the result
    plt.figure(figsize=(10, 8))
    plt.scatter(transformed_vectors[:, 0], transformed_vectors[:, 1], marker='.', s=1)
    plt.title("Spectral Embedding Visualization")
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.show()



from sklearn.datasets import make_swiss_roll

def spectral_embedding_visualization_swiss(n_samples=1000, n_components=2, eigen_solver='amg'):
    # Generate the Swiss roll dataset
    data, _ = make_swiss_roll(n_samples)

    # Create a Spectral Embedding object and fit-transform the data
    embedding = SpectralEmbedding(n_components=n_components, eigen_solver=eigen_solver)
    transformed_data = embedding.fit_transform(data)

    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], marker='.', s=1)
    plt.title("Spectral Embedding Visualization of Swiss Roll")
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.show()

