import random
import time
import matplotlib.pyplot as plt
import cupy as cp
from cuml import UMAP
from cuml.metrics import trustworthiness
from sklearn.datasets import make_swiss_roll
import cudf

class UMAPVisualizer:
    """
    A class used to visualize high-dimensional data using UMAP (Uniform Manifold Approximation and Projection).

    Attributes
    ----------
    embedding_n_neighbors : int
        The number of neighbors for the UMAP embedding (default is 15)
    pairwise_distances_sample_size : int
        The sample size for pairwise distances (default is 5000)
    elapsed_time : float
        The time taken to fit and transform the data
    transformed_vectors : array
        The transformed vectors after applying UMAP
    avg_distance_umap : float
        The average distance in the UMAP space
    avg_distance_data : float
        The average distance in the original data space
    tw : float
        The trustworthiness of the transformed vectors
    title : str
        The title of the plot
    isSwiss : bool
        A flag to indicate if the data is the Swiss roll dataset

    Methods
    -------
    _fit_transform(word_vectors):
        Fits the UMAP model and transforms the input data.
    plot():
        Plots the transformed vectors.
    word2vec(model, subset_size=None):
        Fits and transforms the word vectors from a Word2Vec model.
    swiss(n_samples):
        Fits and transforms the Swiss roll dataset.
    plot_swiss():
        Plots the original Swiss roll dataset.

    """


    def __init__(self, embedding_n_neighbors=15, pairwise_distances_sample_size=5000):
        """
        Constructs all the necessary attributes for the UMAPVisualizer object.

        Parameters
        ----------
            embedding_n_neighbors : int, optional
                The number of neighbors for the UMAP embedding (default is 15)
            pairwise_distances_sample_size : int, optional
                The sample size for pairwise distances (default is 5000)
        """
         
        self.embedding_n_neighbors = embedding_n_neighbors
        self.pairwise_distances_sample_size = pairwise_distances_sample_size
        self.elapsed_time = None
        self.transformed_vectors = None
        self.avg_distance_umap = None
        self.avg_distance_data = None
        self.tw = None
        self.title = None
        self.isSwiss=bool()

    def _fit_transform(self, word_vectors):
        """
        Fits the UMAP model and transforms the input data.

        Parameters
        ----------
            word_vectors : array
                The word vectors to be transformed
        """

        umap = UMAP(n_neighbors=self.embedding_n_neighbors, n_components=2)
        start_time = time.time()
        self.transformed_vectors = umap.fit_transform(word_vectors)
        self.elapsed_time = time.time() - start_time

        if not self.isSwiss:
            self.tw = trustworthiness(word_vectors.get(), self.transformed_vectors.get(), n_neighbors=5, metric='euclidean')
        else:
            self.tw = trustworthiness(word_vectors, self.transformed_vectors.to_pandas().values, n_neighbors=5, metric='euclidean')

    
    def plot(self):
        """
        Plots the transformed vectors.
        """

        plt.figure(figsize=(10, 8))
        
        # Check if the number of samples is greater than 10,000
        num_samples = self.transformed_vectors.shape[0] if not self.isSwiss else len(self.transformed_vectors)
        if num_samples > 10000:
            indices = random.sample(range(num_samples), 10000)
        else:
            indices = range(num_samples)

        if not self.isSwiss:
            self.transformed_vectors = cp.asnumpy(self.transformed_vectors)
            subset_transformed_vectors = self.transformed_vectors[indices]
            plt.scatter(subset_transformed_vectors[:, 0], subset_transformed_vectors[:, 1], marker='.', s=1)
        else:
            df = self.transformed_vectors.to_pandas().iloc[indices]
            plt.scatter(*df.values.T, s=10, c=self.swissroll_labels[df.index], cmap="Spectral", alpha=0.5)

        plt.text(0.99, 0.01, "{:.2f} s".format(self.elapsed_time), transform=plt.gca().transAxes, size=14, horizontalalignment="right")
        plt.title(self.title)
        plt.xlabel("Embedding Dimension 1")
        plt.ylabel("Embedding Dimension 2")
        plt.show()



    def word2vec(self, model, subset_size=None):
        """
        Fits and transforms the word vectors from a Word2Vec model.

        Parameters
        ----------
            model : gensim.models.Word2Vec
                The Word2Vec model
            subset_size : int, optional
                The size of the subset of word vectors to be used (default is None, which means all word vectors are used)
        """

        self.isSwiss=False
        all_words = list(model.key_to_index.keys())
        if subset_size is not None:
            all_words = random.sample(all_words, subset_size)

        word_vectors = cp.asarray([model[word] for word in all_words], dtype=cp.float32)

        self._fit_transform(word_vectors)
        self.title = "UMAP Visualization of Word2Vec (10K Samples)"
        print("Completed for Word2Vec")

    def swiss(self, n_samples):
        """
        Fits and transforms the Swiss roll dataset.

        Parameters
        ----------
            n_samples : int
                The number of samples in the Swiss roll dataset
        """

        self.isSwiss = True
        # Generate the Swiss roll dataset
        self.swissroll, self.swissroll_labels = make_swiss_roll(n_samples=n_samples, noise=0.1, random_state=42)

        gdf = cudf.DataFrame.from_records(self.swissroll) # Convert the data to a cudf dataframe

        self._fit_transform(gdf)
        self.title = "UMAP Visualization of Swiss Roll (10K Samples)"
        print("Completed for SwissRoll")



    def plot_swiss(self):
        """
        Plots the original Swiss roll dataset.
        """

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Check if the number of samples is greater than 10,000
        num_samples = self.swissroll.shape[0]
        if num_samples > 10000:
            indices = random.sample(range(num_samples), 10000)
        else:
            indices = range(num_samples)

        subset_swissroll = self.swissroll[indices]
        subset_swissroll_labels = self.swissroll_labels[indices]

        ax.scatter(subset_swissroll[:, 0], subset_swissroll[:, 1], subset_swissroll[:, 2], c=subset_swissroll_labels, cmap=plt.cm.Spectral)
        ax.set_title("Original Swiss Roll (10K Samples)")
        plt.show()


