#Follow the link for setting up cudf and cuml libraries with wsl2:  https://docs.rapids.ai/install#wsl2-pip

import cupy as cp
from cuml.manifold import UMAP
import matplotlib.pyplot as plt
import random
from sklearn.datasets import make_swiss_roll


def umap_visualization(model, subset_size=None, n_neighbors=15, n_components=2):
    # Get all words from the vocabulary
    all_words = list(model.key_to_index.keys())

    # If a subset size is specified, randomly sample the words
    if subset_size is not None:
        all_words = random.sample(all_words, subset_size)

    # Create a generator that yields word vectors one at a time
    def word_vector_generator():
        for word in all_words:
            yield model[word]

    word_vectors = word_vector_generator()

    # Apply UMAP on all words
    umap = UMAP(n_neighbors=n_neighbors, n_components=n_components)
    transformed_vectors = umap.fit_transform(cp.asarray(list(word_vectors), dtype=cp.float32))

    # Convert the transformed vectors to numpy arrays for plotting
    transformed_vectors = cp.asnumpy(transformed_vectors)

    # Create a scatter plot with small dots
    plt.figure(figsize=(10, 8))
    plt.scatter(transformed_vectors[:, 0], transformed_vectors[:, 1], marker='.', s=1)

    plt.title("UMAP Visualization")
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.show()





def umap_visualization_swiss(n_samples=1000, n_neighbors=15, n_components=2):
    # Generate the Swiss roll dataset
    data, _ = make_swiss_roll(n_samples)

    # Convert the data to a cupy array
    data_cp = cp.asarray(data, dtype=cp.float32)

    # Apply UMAP
    umap = UMAP(n_neighbors=n_neighbors, n_components=n_components)
    transformed_data = umap.fit_transform(data_cp)

    # Convert the transformed data to numpy arrays for plotting
    transformed_data_np = cp.asnumpy(transformed_data)

    # Create a scatter plot with small dots
    plt.figure(figsize=(10, 8))
    plt.scatter(transformed_data_np[:, 0], transformed_data_np[:, 1], marker='.', s=1)

    plt.title("UMAP Visualization of Swiss Roll")
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.show()
