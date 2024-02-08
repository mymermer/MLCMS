import random
import time
import matplotlib.pyplot as plt
import cupy as cp
from cuml import UMAP
from cuml.metrics import trustworthiness
from sklearn.datasets import make_swiss_roll
import cudf

class UMAPVisualizer:
    def __init__(self, embedding_n_neighbors=15, pairwise_distances_sample_size=5000):
        self.embedding_n_neighbors = embedding_n_neighbors
        self.pairwise_distances_sample_size = pairwise_distances_sample_size
        self.elapsed_time = None
        self.transformed_vectors = None
        self.avg_distance_umap = None
        self.avg_distance_data = None
        self.tw = None
        self.title = None
        self.isSwiss=bool()

    def fit_transform(self, word_vectors):
        umap = UMAP(n_neighbors=self.embedding_n_neighbors, n_components=2)
        start_time = time.time()
        self.transformed_vectors = umap.fit_transform(word_vectors)
        self.elapsed_time = time.time() - start_time

        if not self.isSwiss:
            self.tw = trustworthiness(word_vectors.get(), self.transformed_vectors.get(), n_neighbors=5, metric='euclidean')
        else:
            self.tw = trustworthiness(word_vectors, self.transformed_vectors.to_pandas().values, n_neighbors=5, metric='euclidean')

    
    def plot(self):

        plt.figure(figsize=(10, 8))
        
        if not self.isSwiss:
            self.transformed_vectors = cp.asnumpy(self.transformed_vectors)

            plt.scatter(self.transformed_vectors[:, 0], self.transformed_vectors[:, 1], marker='.', s=1)
        else:
            plt.scatter(*self.transformed_vectors.to_pandas().values.T, s=10, c=self.swissroll_labels, cmap="Spectral", alpha=0.5)

        plt.text(0.99, 0.01, "{:.2f} s".format(self.elapsed_time), transform=plt.gca().transAxes, size=14, horizontalalignment="right")
        plt.title(self.title)
        plt.xlabel("Embedding Dimension 1")
        plt.ylabel("Embedding Dimension 2")
        plt.show()

    def word2vec(self, model, subset_size=None):
        self.isSwiss=False
        all_words = list(model.key_to_index.keys())
        if subset_size is not None:
            all_words = random.sample(all_words, subset_size)

        word_vectors = cp.asarray([model[word] for word in all_words], dtype=cp.float32)

        self.fit_transform(word_vectors)
        self.title = "UMAP Visualization of Word2Vec"
        print("Completed for Word2Vec")

    def swiss(self, n_samples):
        self.isSwiss = True
        # Generate the Swiss roll dataset
        self.swissroll, self.swissroll_labels = make_swiss_roll(n_samples=n_samples, noise=0.1, random_state=42)

        gdf = cudf.DataFrame.from_records(self.swissroll) # Convert the data to a cudf dataframe

        self.fit_transform(gdf)
        self.title = "UMAP Visualization of Swiss Roll"
        print("Completed for SwissRoll")


    def plot_swiss(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.swissroll[:, 0], self.swissroll[:, 1], self.swissroll[:, 2], c=self.swissroll_labels, cmap=plt.cm.Spectral)
        ax.set_title("Original Swiss Roll")
        plt.show()

