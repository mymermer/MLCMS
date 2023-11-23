#in the terminal 
#git clone https://gitlab.com/datafold-dev/datafold.git
#cd datafold
#& C:/Users/AhnNayeon/AppData/Local/Programs/Python/Python39/python.exe setup.py install



import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D  # noqa: F401
import numpy as np
from sklearn import datasets, manifold
from sklearn.datasets import make_swiss_roll
from Diffusion_Map import diffusion_map, create_dataset_subtask1, visualize_dataset, plot_eigenValues

import datafold.dynfold as dfold
import datafold.pcfold as pfold
from datafold.dynfold import LocalRegressionSelection
from datafold.utils.plot import plot_pairwise_eigenvector



#---Generate Swiss roll data set---

nr_samples = 15000
rng = np.random.default_rng(1)

# reduce number of points for plotting
nr_samples_plot = 1000
idx_plot = rng.permutation(nr_samples)[0:nr_samples_plot]


X, X_color = make_swiss_roll(nr_samples, random_state=3, noise=0)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    X[idx_plot, 0],
    X[idx_plot, 1],
    X[idx_plot, 2],
    c=X_color[idx_plot],
    cmap=plt.cm.Spectral,
)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Swiss Roll Dataset")
ax.view_init(10, 70)
plt.show()
fig.savefig("C:/Users/AhnNayeon/Downloads/swiss_roll.png")

#manifold 
X_pcm = pfold.PCManifold(X)
X_pcm.optimize_parameters()

print(f"epsilon={X_pcm.kernel.epsilon}, cut-off={X_pcm.cut_off}")

# # Diffusion map processing
# lambda_l, phi_l = diffusion_map(X, 5)

# # Plotting the eigenvalues

# plot_eigenValues(phi_l, lambda_l, 5)

# # Replace 'unfolded_data' with the correct output from the diffusion map
# # Usually, you might want to use the first two non-trivial eigenvectors
# unfolded_data = phi_l[:, 1:3]  # This is a placeholder, adjust as needed

# # Visualization of the unfolded data
# plt.scatter(unfolded_data[:, 0], unfolded_data[:, 1], c=X_color[idx_plot], cmap=plt.cm.Spectral)
# plt.title("Unfolded Swiss Roll")
# plt.xlabel("Dimension 1")
# plt.ylabel("Dimension 2")
# plt.show()
# fig.savefig("C:/Users/AhnNayeon/Downloads/swiss_roll_2.png")
