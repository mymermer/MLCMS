from scipy.spatial.distance import pdist,squareform
import numpy as np
from scipy.linalg import sqrtm

def diffusion_map(data:"np.ndarray",L : int ,epsilon :float =0.05):
    distances=pdist(data)
    diameter = np.max(distances)
    epsilon = epsilon * diameter 
    distance_matrix_D = squareform(distances)
    
    
    kernel_matrix_W = np.exp(-distance_matrix_D**2/epsilon**2)  # kernel matrix W_ij
    diagonal_normalization_matrix_P = np.diag(np.sum( kernel_matrix_W,axis=1)) # diagonal normalization matrix P_ii



    kernel_matrix_K=np.linalg.inv(diagonal_normalization_matrix_P) @ kernel_matrix_W @ np.linalg.inv(diagonal_normalization_matrix_P) #Normalize W to form the kernel matrix K
    diagonal_normalization_matrix_Q = np.diag(np.sum( kernel_matrix_K,axis=1)) #diagonal normalization matrix Q_ii
    Q_1_2=sqrtm( np.linalg.inv(diagonal_normalization_matrix_Q)) 
    symmetric_matrix_T= Q_1_2 @ kernel_matrix_K @ Q_1_2
    
    
    eigenvalues , eigenvectors = np.linalg.eig(symmetric_matrix_T)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    eigenvalues=eigenvalues[: L+1]
    eigenvectors=eigenvectors[:,:L+1]
    λ=np.sqrt(eigenvalues **(1/epsilon)) #eigen value
    Φ=Q_1_2 @ eigenvectors #eigen vector

    return λ, Φ


