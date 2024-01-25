import matplotlib.pyplot as plt
from matplotlib import ticker
from sklearn import datasets, manifold


# Generate swiss roll dataset (input data)
sr_points, sr_color = datasets.make_swiss_roll(n_samples=10000, random_state=0) 

# Plot the original Swiss Roll in 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
fig.add_axes(ax)
ax.scatter(
    sr_points[:, 0], sr_points[:, 1], sr_points[:, 2], c=sr_color, s=50, alpha=0.8
)
ax.set_title("Swiss Roll in Ambient Space")
ax.view_init(azim=-66, elev=12)
_ = ax.text2D(0.8, 0.05, s="n_samples=10000", transform=ax.transAxes)

# Apply Spectral_Embedding method from sklearn to the Swiss Roll dataset

params = {
    # "n_neighbors": n_neighbors,
    "n_components": 2, # embed into 2 dimensions
    "affinity" : "rbf", # Gaussian kernel (aka Radial Basis Function (RBF) or heat kernel) affinity ie radius based neighburhood 
    "eigen_solver": "lobpcg", # Locally-Optimized Block-Preconditioned Conjugate Gradient (lobpcg) eigensolver
    "random_state": 0,
}

spectral  = manifold.SpectralEmbedding(**params)
sr_spectral = spectral.fit_transform(sr_points)

# Plot the embedded Swiss Roll in 2D
def plot_2d(points, points_color, title):
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, points_color)
    plt.show()

def add_2d_scatter(ax, points, points_color, title=None):
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

plot_2d(sr_spectral, sr_color, "Spectral Embedding")



plt.show()