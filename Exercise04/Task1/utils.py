from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import eig
def plot_phase_portrait(A, title):
    w = 1
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]
    x = np.array([X, Y])

    U = A[0, 0] * X + A[0, 1] * Y
    V = A[1, 0] * X + A[1, 1] * Y

    eigenvalues, _ = eig(A)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    # Plot streamlines
    axs[1].set_title(title)
    strm = axs[1].streamplot(X, Y, U, V, color='k', linewidth=2)

    # Plot eigenvalues in the complex plane
    axs[0].scatter(np.real(eigenvalues), np.imag(eigenvalues), color='red', label='Eigenvalues')
    axs[0].axhline(0, color='black', linewidth=2)
    axs[0].axvline(0, color='black', linewidth=2)
    axs[0].set_xlabel('Real Part')
    axs[0].set_ylabel('Imaginary Part')
    axs[0].legend()
    axs[0].grid(True)

    plt.tight_layout()
    plt.show()