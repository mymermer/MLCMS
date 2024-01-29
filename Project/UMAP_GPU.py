#UMAP with GPU acceleration
#Follow the link for setting up cudf and cuml libraries with wsl2:  https://docs.rapids.ai/install#wsl2-pip

def umap_visualization(model, subset_size=20000, n_neighbors=15, n_components=2):
    import cudf
    from cuml.manifold import UMAP
    import matplotlib.pyplot as plt
    import random

    # Get a random subset of words from the vocabulary
    subset_words = random.sample(list(model.key_to_index.keys()), subset_size)

    # Get the word vectors for the subset and convert it to a cuDF DataFrame
    subset_word_vectors = cudf.DataFrame([model[word] for word in subset_words])

    # Apply UMAP on the subset
    umap = UMAP(n_neighbors=n_neighbors, n_components=n_components)
    transformed_vectors = umap.fit_transform(subset_word_vectors)

    # Create a scatter plot with small dots
    plt.figure(figsize=(10, 8))
    plt.scatter(transformed_vectors[0].to_pandas().values, transformed_vectors[1].to_pandas().values, marker='.', s=1)

    plt.title("UMAP Visualization")
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.show()
