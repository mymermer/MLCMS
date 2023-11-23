import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.datasets import face
from skimage.transform import resize



# ----- 1st Part -----

def read_2d_data(file_path):
    # Reads data from a given file path and returns two lists, x and y
    x, y = [], []
    with open(file_path, "r") as file:
        for line in file:
            if line.strip():  # Ignore empty lines
                parts = line.split()  # Splitting by whitespace
                x.append(float(parts[0]))
                y.append(float(parts[1]))
    return x, y

def perform_pca(data, components):
    # Performs PCA on the data and returns the PCA object and transformed data
    pca = PCA(n_components=components)
    transformed_data = pca.fit_transform(data)
    return pca, transformed_data

def plot_data_with_pca(x, y, pca, data, output_path):
    # Plots the original data and PCA components
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.5)
    plt.title('Dataset and Principal Components')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)

    origin = np.mean(data, axis=0)  # Origin point for the PCA vectors
    for length, vector in zip(pca.explained_variance_, pca.components_):
        v = vector * 3 * np.sqrt(length)
        plt.quiver(*origin, *v, scale=1, scale_units='xy', angles='xy', color='r')

    plt.savefig(output_path, format='png', dpi=300)
    plt.show()

def first_task(file_path, output_path):
    x, y = read_2d_data(file_path)
    data = np.column_stack((x, y))
    pca, transformed_data = perform_pca(data, 2)
    plot_data_with_pca(x, y, pca, data, output_path)
    variance_explained = pca.explained_variance_ratio_

    return variance_explained, pca, transformed_data

file_path = "C:/Users/AhnNayeon/Downloads/pca_dataset.txt"
output_path = "C:/Users/AhnNayeon/Downloads/ex3task1-1.png"

energy1, pca, transformed_data = first_task(file_path, output_path)
overall_energy1 = np.sum(energy1) if energy1 is not None else None

if energy1 is not None:
    print(f'For part 1, each component explains {energy1}')
    print(f'The overall energy would be {overall_energy1}')

# ----- 2nd Part -----


def get_image():
    # Gets an image file and converts it to gray
    image = face(gray=True)
    image_resized = resize(image, (249, 185))
    plt.imshow(image_resized, cmap='gray')
    plt.title('Part 2 Original Image')
    plt.axis('off')
    plt.show()
    return image_resized

def apply_pca_and_reconstruct(image, num_components):
    # Applies PCA to the given image and reconstructs it
    image_flattened = image.reshape(image.shape[0], -1)
    pca = PCA(n_components=num_components)
    transformed = pca.fit_transform(image_flattened)
    reconstructed = pca.inverse_transform(transformed)
    reconstructed_image = reconstructed.reshape(image.shape)
    variance_explained = np.sum(pca.explained_variance_ratio_)
    return reconstructed_image, variance_explained

def second_part(image, output_path):
    max_components = min(image.shape)
    components = [max_components, 120, 50, 10]
    reconstructed_images = []
    variances_explained = []

    for n in components:
        reconstructed_image, variance_explained = apply_pca_and_reconstruct(image, n)
        reconstructed_images.append(reconstructed_image)
        variances_explained.append(variance_explained)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    for i, img in enumerate(reconstructed_images, 2):
        plt.subplot(2, 3, i)
        plt.imshow(img, cmap='gray')
        plt.title(f'{components[i-2]} Components\nEnergy: {variances_explained[i-2]:.5f}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, format='png', dpi=300)  
    plt.show()

image_resized = get_image()
output_path = "C:/Users/AhnNayeon/Downloads/ex3task1-2.png" 
second_part(image_resized, output_path)


#-----3rd Part-----

# Load the data
data3 = pd.read_csv('C:/Users/AhnNayeon/Downloads/data_DMAP_PCA_vadere.txt', sep=' ')

def plot_pedestrian_path(data, pedestrian_index, save_path):
    
    x_column = 2 * pedestrian_index - 2  
    y_column = 2 * pedestrian_index - 1  

    x_coordinates = data.iloc[:, x_column]
    y_coordinates = data.iloc[:, y_column]

    plt.figure(figsize=(10, 6))
    plt.plot(x_coordinates, y_coordinates)
    plt.title(f"Trajectory of Pedestrian {pedestrian_index}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()
     

plot_pedestrian_path(data3, 1, 'C:/Users/AhnNayeon/Downloads/ex3task1-3-1.png')
plot_pedestrian_path(data3, 2, 'C:/Users/AhnNayeon/Downloads/ex3task1-3-2.png')

# Perform PCA with 2 components
pca_2 = PCA(n_components=2)  
transformed_data_2 = pca_2.fit_transform(data3)
energy2 = pca_2.explained_variance_ratio_
overall_energy2 = np.sum(energy2)
print(f"For part3, PCA with 2 components, each component explains: {energy2}")
print(f"The overall explained variance is: {overall_energy2}")

# Increase the number of components if necessary
required_components = 2  # Starting with 2 components
if overall_energy2 < 0.9:
    for n_components in range(3, 31):
        pca_n = PCA(n_components=n_components)
        pca_n.fit(data3)
        cumulative_variance = np.cumsum(pca_n.explained_variance_ratio_)
        if cumulative_variance[-1] >= 0.9:
            required_components = n_components
            print(f"Number of components needed to capture >90% of the variance: {n_components}")
            break

# Perform PCA with the required number of components
pca_final = PCA(n_components=required_components)  
transformed_data_final = pca_final.fit_transform(data3)
energy_final = pca_final.explained_variance_ratio_
overall_energy_final = np.sum(energy_final)
print(f"For {required_components} components, each component explains: {energy_final}")
print(f"The overall explained variance is: {overall_energy_final}")