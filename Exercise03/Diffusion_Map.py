from scipy.spatial.distance import pdist,squareform
import numpy as np
from scipy.linalg import sqrtm, eigh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def diffusion_map(data:"np.ndarray",L : int ,epsilon :float =0.05):
    distances=pdist(data)
    diameter = np.max(distances)
    epsilon = epsilon * diameter
    distance_matrix_D = squareform(distances)


    kernel_matrix_W = np.exp(-distance_matrix_D**2/epsilon)  # kernel matrix W_ij
    diagonal_normalization_matrix_P = np.diag(np.sum( kernel_matrix_W,axis=1)) # diagonal normalization matrix P_ii


    kernel_matrix_K=np.linalg.inv(diagonal_normalization_matrix_P) @ kernel_matrix_W @ np.linalg.inv(diagonal_normalization_matrix_P) #Normalize W to form the kernel matrix K
    diagonal_normalization_matrix_Q = np.diag(np.sum( kernel_matrix_K,axis=1)) #diagonal normalization matrix Q_ii
    Q_1_2=sqrtm( np.linalg.inv(diagonal_normalization_matrix_Q))
    symmetric_matrix_T= Q_1_2 @ kernel_matrix_K @ Q_1_2


    eigenvalues , eigenvectors = np.linalg.eigh(symmetric_matrix_T)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    eigenvalues=eigenvalues[: L+1]
    eigenvectors=eigenvectors[:,:L+1]
    λ=np.sqrt(eigenvalues **(1/epsilon)) #eigen value
    Φ=Q_1_2 @ eigenvectors #eigen vector

    return λ, Φ

def create_dataset_subtask1(n:int=1000):
    tk = 2 * np.pi * np.linspace(0, n, n) / (n + 1)
    xk = np.vstack((np.cos(tk), np.sin(tk))).T
    return xk, tk

def visualize_dataset(xk, tk):
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

def plot_eigenValues(tk, phi_val, lambda_val, num=5):
    for i in range(num):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(tk, phi_val[:,i],c=tk, cmap='viridis')
        ax.set_title("EigenValue:{:.4f}".format(lambda_val[i]))
        fig.show()


