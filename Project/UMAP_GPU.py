#Follow the link for setting up cudf and cuml libraries with wsl2:  https://docs.rapids.ai/install#wsl2-pip

import cudf
import cuml
from cuml.manifold import UMAP
from cuml.metrics import pairwise_distances, trustworthiness
import cupy as cp
import matplotlib.pyplot as plt
import random
import time
from sklearn import datasets



def umap_visualization_word2vec(model, subset_size=None, embedding_n_neighbors=15, pairwise_distances_sample_size=5000):
    all_words = list(model.key_to_index.keys())
    if subset_size is not None:
        all_words = random.sample(all_words, subset_size)

    word_vectors = cp.asarray([model[word] for word in all_words], dtype=cp.float32)

    umap = UMAP(n_neighbors=embedding_n_neighbors, n_components=2)
    start_time = time.time()
    transformed_vectors = umap.fit_transform(word_vectors)
    elapsed_time = time.time() - start_time

    sample_umap = cp.asarray(random.sample(transformed_vectors.tolist(), pairwise_distances_sample_size))
    sample_word_vectors = cp.asarray(random.sample(word_vectors.tolist(), pairwise_distances_sample_size))

    avg_distance_umap = pairwise_distances(sample_umap).flatten().mean()
    avg_distance_word_vectors = pairwise_distances(sample_word_vectors).flatten().mean()
    print(f"Average distance in UMAP embeddings: {avg_distance_umap}")
    print(f"Average distance in Word2Vec dataset: {avg_distance_word_vectors}")

    tw = trustworthiness(word_vectors.get(), transformed_vectors.get(), n_neighbors=5, metric='euclidean')
    print(f'Trustworthiness of the embedding: {tw}')

    transformed_vectors = cp.asnumpy(transformed_vectors)

    plt.figure(figsize=(10, 8))
    plt.scatter(transformed_vectors[:, 0], transformed_vectors[:, 1], marker='.', s=1)
    plt.text(0.99, 0.01, "{:.2f} s".format(elapsed_time), transform=plt.gca().transAxes, size=14, horizontalalignment="right")
    plt.title("UMAP Visualization")
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.show()








def umap_visualization_swiss(n_samples, embedding_n_neighbors, pairwise_distances_sample_size):
    # Generate the Swiss roll dataset
    swissroll, swissroll_labels = datasets.make_swiss_roll(n_samples=n_samples, noise=0.1, random_state=42)

    # Apply UMAP
    reducer = cuml.UMAP(n_neighbors=embedding_n_neighbors, init="spectral", random_state=42, min_dist=0.5)
    gdf = cudf.DataFrame.from_records(swissroll) # Convert the data to a cudf dataframe
    start_time = time.time()
    embedding = reducer.fit_transform(gdf)
    elapsed_time = time.time() - start_time

    # Randomly sample a subset of the data
    sample_umap = embedding.sample(pairwise_distances_sample_size, replace=False)
    sample_swiss = gdf.sample(pairwise_distances_sample_size, replace=False)

    # Compute pairwise Euclidean distances for the samples
    distances_umap = pairwise_distances(sample_umap)
    distances_swiss = pairwise_distances(sample_swiss)

    # Calculate the average distance
    avg_distance_umap = distances_umap.values.flatten().mean()
    avg_distance_swiss = distances_swiss.values.flatten().mean()
    print(f"Average distance in UMAP embeddings: {avg_distance_umap}")
    print(f"Average distance in Swiss Roll dataset: {avg_distance_swiss}")

    # Evaluate the quality of the embedding with trustworthiness
    tw = cuml.metrics.trustworthiness(swissroll, embedding.to_pandas().values, n_neighbors=5, metric='euclidean')
    print(f'Trustworthiness of the embedding: {tw}')

    # Create a scatter plot with small dots
    plt.figure(figsize=(10, 8))
    plt.scatter(*embedding.to_pandas().values.T, s=10, c=swissroll_labels, cmap="Spectral", alpha=0.5)
    plt.text(0.99, 0.01, "{:.2f} s".format(elapsed_time), transform=plt.gca().transAxes, size=14, horizontalalignment="right")
    plt.title("UMAP Visualization of Swiss Roll")
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.show()
