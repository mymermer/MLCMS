import numpy as np
import matplotlib.pyplot as plt
from scipy.datasets import face
from skimage.transform import resize

def pca_def(data, L, verbose=False):
    
    # Center the matrix by removing the mean
    mean_vector = np.mean(data, axis=0)
    centered_data = data - mean_vector
    
    # Perform Singular Value Decomposition
    U, S, Vt = np.linalg.svd(centered_data, full_matrices=False)

    # Trimming matrices to include only the given number of principal components
    U_reduced = U[:,:L]
    S_reduced = S[:L]
    Vt_reduced = Vt[:L,:]

    # Transform the data
    transformed_data = np.dot(centered_data, Vt_reduced.T)
    variance = np.var(transformed_data, axis=0)

    # Create a diagonal matrix
    S_full = np.zeros((U_reduced.shape[1], Vt_reduced.shape[0]))
    S_full[:L, :L] = np.diag(S_reduced)

    # Reconstruct the data
    reconstructed_data = U_reduced @ S_full @ Vt_reduced + mean_vector

    #Calculating PCA Energy
    pca_energy = np.sum(S_reduced**2)/np.sum(S**2)

    return pca_energy, variance, centered_data, U, S, Vt, transformed_data, reconstructed_data


def plot_data_with_pca(data, explained_variance,Vt,output_path ):
    # Plots the original data and PCA components
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:,0], data[:,1], alpha=0.5)
    plt.title('Dataset and Principal Components')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)

    origin = np.mean(data, axis=0)  # Origin point for the PCA vectors
    for length, vector in zip(explained_variance, Vt):
        v = vector * 3 * np.sqrt(length)
        plt.quiver(*origin, *v, scale=1, scale_units='xy', angles='xy', color='r')

    plt.savefig(output_path, format='png', dpi=300)
    plt.show()


def get_image():
    # Gets an image file and converts it to gray
    image = face(gray=True)
    image_resized = resize(image, (249, 185))
    plt.imshow(image_resized, cmap='gray')
    plt.title('Part 2 Original Image')
    plt.axis('off')
    plt.show()
    return image_resized

def second_part(image, output_path):
    max_components = min(image.shape)
    components = [max_components, 120, 50, 10]
    reconstructed_images = []
    variances_explained = []
    pca_energies = []

    for n in components:
        pca_energy, variance, _, _, _, _, _, reconstructed_data= pca_def(image, n)
        reconstructed_images.append(reconstructed_data)
        variances_explained.append(variance)
        pca_energies.append(pca_energy)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    for i, img in enumerate(reconstructed_images, 2):
        plt.subplot(2, 3, i)
        plt.imshow(img, cmap='gray')
        plt.title(f'{components[i-2]} Components\nEnergy: {pca_energies[i-2]:.5f}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, format='png', dpi=300)  
    plt.show()

