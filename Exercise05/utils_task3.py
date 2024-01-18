import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq

def linear_approximation(A, x0, delta_t):
    """
   Computes the linear approximation of a state vector after a time step.

   Parameters:
   - A (numpy.ndarray): Linear transformation matrix.
   - x0 (numpy.ndarray): Initial state vector.
   - delta_t (float): Time step for the approximation.

   Returns:
    numpy.ndarray: Linearly approximated state vector after the specified time step.
   """
    return x0 + delta_t * np.dot(A, x0.T)

def nonlinear_approximation(v_x1, x0, delta_t):
    """
    Computes the nonlinear approximation of a state vector after a time step.

    Parameters:
    - v_x1 (numpy.ndarray): Vector representing the derivative of the state vector.
    - x0 (numpy.ndarray): Initial state vector.
    - delta_t (float): Time step for the approximation.

    Returns:
    numpy.ndarray: Nonlinearly approximated state vector after the specified time step.
    """
    return x0 + delta_t * v_x1

def scatter_plot(a, b, a_label, b_label):
    """
    Generates a scatter plot for two sets of data points.

    Parameters:
    - a (numpy.ndarray): Data points for the first set, where each row represents a point.
    - b (numpy.ndarray): Data points for the second set, where each row represents a point.
    - a_label (str): Label for the first set in the legend.
    - b_label (str): Label for the second set in the legend.
    """
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(a[:, 0], a[:, 1], label=a_label)
    plt.scatter(b[:, 0], b[:, 1], label=b_label)
    plt.legend()
    plt.show()

def rbf(data, l, eps):
    """
    Computes the Radial Basis Function (RBF) features for a given dataset.

    Parameters:
    - data (numpy.ndarray): Input data matrix with each row representing a data point.
    - l (int): Number of radial basis functions (RBFs) to use.
    - eps (float): Radial basis function width parameter.

    Returns:
    numpy.ndarray: Matrix of RBF features with shape (number of data points, number of RBFs).
    """
    phi = []
    np.random.seed(42)
    x_l = np.random.choice(len(data), l, replace=False)

    for i in range(l):
        normalized_values = data - data[x_l[i]]
        phi_l = np.exp(-np.sum(normalized_values ** 2, axis=1) / eps ** 2)
        phi.append(phi_l)

    return np.array(phi).T  # Transpose to have shape (number of data points, number of RBFs)

def rbf_approximation(phi, coefficients):
    """
    Computes the approximation using Radial Basis Function (RBF) features and coefficients.

    Parameters:
    - phi (numpy.ndarray): Matrix of RBF features with shape (number of data points, number of RBFs).
    - coefficients (numpy.ndarray): Coefficients for the linear combination of RBF features.

    Returns:
    numpy.ndarray: Approximated values using the RBF features and coefficients.
    """
    return np.dot(phi, coefficients)

def find_optimal_rbf_setting(larray, epsarray, data, v_hat):
    """
    Finds the optimal Radial Basis Function (RBF) settings that minimize Mean Squared Error (MSE).

    Parameters:
    - larray (list): List of potential values for the number of RBFs.
    - epsarray (list): List of potential values for the RBF width parameter.
    - data (numpy.ndarray): Input data matrix with each row representing a data point.
    - v_hat (numpy.ndarray): Target variable values to be approximated.

    Returns:
    tuple: Tuple containing:
        - list: List of tuples containing (l, eps, MSE) for each combination of RBF settings.
        - float: Minimum MSE achieved.
        - float: Optimal RBF width parameter.
        - int: Optimal number of RBFs.
    """
    mse_rbf_array = []
    mse_rbf_least = 10
    for l in larray:
        for eps in epsarray:

            phi = rbf(data, l, eps)

            # Perform linear regression to estimate coefficients
            coefficients = lstsq(phi, v_hat, cond=None)[0]

            # Evaluate RBF approximation for all data points
            rbf_approximated_v = np.array(rbf_approximation(phi, coefficients))

            # Calculate Mean Squared Error
            mse_rbf = np.mean(np.square(v_hat - rbf_approximated_v))
            mse_rbf_array.append((l, eps, mse_rbf))

            # Update if the current combination yields lower MSE
            if mse_rbf < mse_rbf_least:
                mse_rbf_least = mse_rbf
                eps_least = eps
                l_least = l

    return mse_rbf_array, mse_rbf_least, eps_least, l_least

def plot_3d(array):
    """
   Creates a 3D scatter plot based on the input array.

   Parameters:
   - array (list): List of tuples containing (l, eps, mse_rbf) values for each RBF setting.
    """
    l, eps, mse_rbf = zip(*array)

    # Convert to numpy arrays
    l = np.array(l)
    eps = np.array(eps)
    mse_rbf = np.array(mse_rbf)

    # Get unique values in l and eps
    unique_l = np.unique(l)
    unique_eps = np.unique(eps)

    # Reshape to 2D arrays
    l = l.reshape((len(unique_eps), len(unique_l)))
    eps = eps.reshape((len(unique_eps), len(unique_l)))
    mse_rbf = mse_rbf.reshape((len(unique_eps), len(unique_l)))

    # Create a 3D plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points with colors based on mse_rbf values
    scatter = ax.scatter(l, eps, mse_rbf, c=mse_rbf, cmap='viridis', marker='o')

    # Set labels
    ax.set_xlabel('l value')
    ax.set_ylabel('eps value')
    ax.set_zlabel('mse_rbf')

    # Show the plot
    plt.show()

def radial_basis_function_vectorized(X, Y, l, eps, data):
    """
    Computes the radial basis function (RBF) values for a grid of points in a vectorized manner.

    Parameters:
    - X (numpy.ndarray): X-coordinates of the grid points.
    - Y (numpy.ndarray): Y-coordinates of the grid points.
    - l (int): Number of radial basis functions (RBFs) to use.
    - eps (float): Radial basis function width parameter.
    - data (numpy.ndarray): Input data matrix with each row representing a data point.

    Returns:
    numpy.ndarray: 3D array of RBF values for each point in the grid.
    """
    phi = np.zeros((len(X), len(Y), l))

    np.random.seed(42)
    x_l = np.random.choice(len(data), l, replace=False)

    for i in range(l):
        normalized_values_x = X - data[x_l[i], 0]
        normalized_values_y = Y - data[x_l[i], 1]

        phi[:, :, i] = np.exp(-(normalized_values_x**2 + normalized_values_y**2) / eps**2)

    return phi

def phase_portrait(l, eps, data, coefficients):
    """
    Generates a phase portrait using radial basis function (RBF) values and coefficients.

    Parameters:
    - l (int): Number of radial basis functions (RBFs) used.
    - eps (float): Radial basis function width parameter.
    - data (numpy.ndarray): Input data matrix with each row representing a data point.
    - coefficients (numpy.ndarray): Coefficients for the linear combination of RBF features.
    """
    # Generate a grid of points
    x = np.linspace(-4.5, 4.5)
    y = np.linspace(-4.5, 4.5)
    X, Y = np.meshgrid(x, y)

    # Calculate the radial basis function values for the grid

    phi_values = radial_basis_function_vectorized(X, Y, l, eps, data)

    # Compute the vector field
    U = np.sum(coefficients[:, 0] * phi_values, axis=2)
    V = np.sum(coefficients[:, 1] * phi_values, axis=2)

    # Plot the streamplot
    plt.streamplot(X, Y, U, V, density=1, linewidth=1, arrowsize=1, cmap='autumn')
    plt.show()