from scipy.spatial.distance import pdist,squareform
import numpy as np
from scipy.linalg import sqrtm, eigh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def diffusion_map(data: np.ndarray, L: int, epsilon: float = 0.05):
    """
    Perform diffusion mapping on the given data.

    Parameters:
    - data (np.ndarray): Input data matrix with shape (n_samples, n_features).
    - L (int): Number of eigenvalues and corresponding eigenvectors to compute.
    - epsilon (float): Diffusion parameter controlling the scale of the Gaussian kernel. Default is 0.05.

    Returns:
    - λ (np.ndarray): Array of eigenvalues.
    - Φ (np.ndarray): Matrix of corresponding eigenvectors.

    Note:
    - The function computes the diffusion map eigenvalues (λ) and eigenvectors (Φ) using the given data.
    - The diffusion map captures the intrinsic geometry and structure of the data.
    """

    # Calculate pairwise distances
    distances = pdist(data)
    diameter = np.max(distances)
    epsilon = epsilon * diameter

    # Form distance matrix
    distance_matrix_D = squareform(distances)

    # Construct the kernel matrix W
    kernel_matrix_W = np.exp(-distance_matrix_D**2 / epsilon)

    # Diagonal normalization matrix P_ii
    diagonal_normalization_matrix_P = np.diag(np.sum(kernel_matrix_W, axis=1))

    # Normalize W to form the kernel matrix K
    kernel_matrix_K = np.linalg.inv(diagonal_normalization_matrix_P) @ kernel_matrix_W @ np.linalg.inv(diagonal_normalization_matrix_P)

    # Diagonal normalization matrix Q_ii
    diagonal_normalization_matrix_Q = np.diag(np.sum(kernel_matrix_K, axis=1))

    # Form the symmetric matrix T
    Q_1_2 = sqrtm(np.linalg.inv(diagonal_normalization_matrix_Q))
    symmetric_matrix_T = Q_1_2 @ kernel_matrix_K @ Q_1_2

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric_matrix_T)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Select the top L eigenvalues and corresponding eigenvectors
    eigenvalues = eigenvalues[:L+1]
    eigenvectors = eigenvectors[:, :L+1]

    # Compute the diffusion map components
    λ = np.sqrt(eigenvalues ** (1/epsilon))
    Φ = Q_1_2 @ eigenvectors

    return λ, Φ



def create_dataset_subtask1(n:int=1000):
    """
    Generating the dataset for subtask 1

    Parameters:
    - n (int): Number of samples to be generated.

    Returns:
    - xk (np.ndarray): Array of shape (n, 2) containing the generated samples.
    - tk (np.ndarray): Array of shape (n, ) containing the corresponding angles.

    """
    # Generating an array of n values evenly spaced between 0 and n and
    tk = 2 * np.pi * np.linspace(0, n, n) / (n + 1)
    xk = np.vstack((np.cos(tk), np.sin(tk))).T
    return xk, tk

def visualize_dataset(xk, tk):
    """
    Visualizing the dataset
    
    Parameters:
    - xk (np.ndarray): Array of shape (n, 2) containing the generated samples.
    - tk (np.ndarray): Array of shape (n, ) containing the corresponding angles.

    """
    fig = plt.figure(figsize=(12, 6))

    #2D Visualization
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(xk[:, 0], xk[:, 1], c=tk, cmap='viridis')
    ax1.set_title('2D Visualization of Dataset')
    ax1.set_xlabel('Cosine Values')
    ax1.set_ylabel('Sine Values')

    # 3D Visualization
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(xk[:, 0], xk[:, 1], tk, c=tk, cmap='viridis')
    ax2.set_title('3D Visualization of Dataset')
    ax2.set_xlabel('Cosine Values')
    ax2.set_ylabel('Sine Values')
    ax2.set_zlabel('Angles (tk)')

    plt.show()

    fig.savefig("Figures/dataset1.png")

def plot_eigenValues(tk, phi_val, num=5):
    """
    
    Parameters:
    - tk (np.ndarray): Array of shape (n, ) containing the corresponding angles.
    - phi_val (np.ndarray): Array of shape (n, L) containing the corresponding eigenvalues.
    - num (int): Number of eigenvalues to be plotted. Default is 5.
    - n (int): Number of data points

    """
    rows = int(np.ceil(num / 3.0))
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten()

    for index, ax in enumerate(axes):
        if index>=num:
            break
        ax.scatter(tk, phi_val[:,index],c=tk, cmap='viridis')
        ax.set_title("Eigenfunction {}".format(index))
        fig.show()
    
    # Removing extra subplots
    for ax in axes[num:]:
        ax.remove()

    plt.tight_layout()

    plt.savefig('Figures/task2.1.png')



def plot_phi_vals(phi_val, L, t, figname):
    """
    Plotting the eigenfunctions
    
    Parameters:
    - phi_val (np.ndarray): Array of shape (n, L) containing the corresponding eigenvalues.
    - L (int): Number of eigenvalues and corresponding eigenvectors to compute.
    - t (np.ndarray): Array of shape (n, ) containing the unrolled positions of the points along the main axis of the Swiss roll.
    - figname (str): Name of the figure to be saved.
    
    """
    rows = int(np.ceil(L / 3.0))
    cols = 3

    eigenvectors = np.around(phi_val[:, :], decimals=10)

    # first non-constant eigenfunction
    phi_one = eigenvectors[:, 1]

    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

    # Flatten the axes array 
    axes = axes.flatten()

    for index, ax in enumerate(axes):
        if index>=L:
            break
        index_label = index if index == 0 else index + 1
        phi_index = eigenvectors[:, index_label]
        ax.scatter(phi_one, phi_index, c=t, cmap="viridis")
        ax.set_title(rf'$\phi 1$ vs $\phi{index_label}$')

    # Removing extra subplots
    for ax in axes[L:]:
        ax.remove()

    plt.tight_layout()

    plt.savefig(figname)