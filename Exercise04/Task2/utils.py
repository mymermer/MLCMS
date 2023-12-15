import numpy as np
import matplotlib.pyplot as plt
from typing import Callable # For type hinting

def plot_bifurcation_diagram(equation: Callable,alpha_initial: float, alpha_end: float, num_of_steps: int, title: str) -> None:
    """
    Plots the bifurcation diagram for the given equation.

    Parameters:
    - equation (callable): The equation to calculate roots.
    - alpha_initial (float): The initial value of alpha.
    - alpha_end (float): The end value of alpha.
    - num_of_steps (int): The number of steps between alpha_initial and alpha_end.
    - title (str): The title of the plot.

    Returns:
    None
    """
    alpha_values = np.linspace(alpha_initial, alpha_end, num_of_steps)

    plt.figure(figsize=(5, 5))
    plt.grid(True, alpha=0.3)
    plt.title(title)
    plt.xlabel('α Values')
    plt.ylabel('Steady States (Real Roots)')

    for alpha in alpha_values:
        result = np.roots(equation(alpha))  # Calculate the roots of the equation
        real_roots = [root.real for root in result if np.isreal(root)]  # Get the real roots

        if real_roots:
            plt.scatter(alpha, 0, s=1, marker='.', c='blue')  # Colorize the x-axis to blue if there is a root
            plt.scatter(alpha, real_roots[0], s=1, marker='.', c='black')

            if len(real_roots) > 1:
                plt.scatter(alpha, real_roots[1], s=1, marker='.', c='black')  # Put 2nd root on the graph if there is any
        else:
            plt.scatter(alpha, 0, s=1, marker='.', c='red')  # Colorize the x-axis to red if there is no root

    plt.xlim(alpha_initial, alpha_end)

    blue_legend = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=5, label='Roots Present')
    red_legend = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=5, label='No Roots')

    plt.legend(handles=[blue_legend, red_legend])

    plt.show()