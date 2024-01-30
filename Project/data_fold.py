import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_s_curve

import datafold.dynfold as dfold
import datafold.pcfold as pfold
from datafold.dynfold import LocalRegressionSelection
import random



def dmap_visualization(word2vec_model, n_samples=15000):
    # Get all words from the vocabulary
    all_words = list(word2vec_model.key_to_index.keys())

    # Randomly sample the words
    all_words = random.sample(all_words, n_samples)

    # Get the vectors from the Word2Vec model
    X = np.array([word2vec_model[word] for word in all_words])
    X_color = np.arange(n_samples)

    # Optimize the parameters of the point cloud manifold
    X_pcm = pfold.PCManifold(X)
    X_pcm.optimize_parameters()

    # Compute the diffusion maps
    dmap = dfold.DiffusionMaps(
        kernel=pfold.GaussianKernel(epsilon=X_pcm.kernel.epsilon, distance=dict(cut_off=X_pcm.cut_off)),
        n_eigenpairs=9,
    ).fit(X_pcm)

    # Select the parsimonious eigenvectors
    selection = LocalRegressionSelection(intrinsic_dim=2, n_subsample=500, strategy="dim").fit(dmap.eigenvectors_)

    # Transform the eigenvectors to the target mapping
    target_mapping = selection.transform(dmap.eigenvectors_)

    # Plot the target mapping
    f, ax = plt.subplots(figsize=(15, 9))
    ax.scatter(target_mapping[:, 0], target_mapping[:, 1], c=X_color, cmap=plt.cm.Spectral)
    ax.set_xlabel("Embedding Dimension 1")
    ax.set_ylabel("Embedding Dimension 2")
    ax.set_title("Diffusion Map Visualization of Word2Vec Vectors")
    plt.show()







def dmap_visualization_s_curve(n_samples=15000):
    rng = np.random.default_rng(1)
    X, X_color = make_s_curve(n_samples, random_state=3, noise=0)
    idx_plot = rng.choice(n_samples, size=1000, replace=False)

    X_pcm = pfold.PCManifold(X)
    X_pcm.optimize_parameters()

    dmap = dfold.DiffusionMaps(
        kernel=pfold.GaussianKernel(epsilon=X_pcm.kernel.epsilon, distance=dict(cut_off=X_pcm.cut_off)),
        n_eigenpairs=9,
    ).fit(X_pcm)

    selection = LocalRegressionSelection(intrinsic_dim=2, n_subsample=500, strategy="dim").fit(dmap.eigenvectors_)

    target_mapping = selection.transform(dmap.eigenvectors_)

    # Plot the target mapping
    f, ax = plt.subplots(figsize=(15, 9))
    ax.scatter(target_mapping[idx_plot, 0], target_mapping[idx_plot, 1], c=X_color[idx_plot], cmap=plt.cm.Spectral)
    ax.set_xlabel("Embedding Dimension 1")
    ax.set_ylabel("Embedding Dimension 2")
    ax.set_title("Diffusion Map Visualization of S-Curve")
    plt.show()
