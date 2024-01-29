import cupy as cp
from cuml.manifold import UMAP
import matplotlib.pyplot as plt
import gc
import random

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
    transformed_vectors = umap.fit_transform(cp.array(list(word_vectors), dtype=cp.float32))

    # Convert the transformed vectors to numpy arrays for plotting
    transformed_vectors = cp.asnumpy(transformed_vectors)

    # Free up memory
    gc.collect()

    # Create a scatter plot with small dots
    plt.figure(figsize=(10, 8))
    plt.scatter(transformed_vectors[:, 0], transformed_vectors[:, 1], marker='.', s=1)

    plt.title("UMAP Visualization")
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.show()
