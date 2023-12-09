import numpy as np
import matplotlib.pyplot as plt

def plot_bifurcation_diagram_equation6(alpha_initial: float=-1, alpha_end: float=1, num_of_steps: int=1001):
    alpha_values = np.linspace(alpha_initial, alpha_end, num_of_steps)

    plt.figure(figsize=(5, 5))
    plt.title('Function 1 Bifurcation Diagram')
    plt.xlabel('Alpha')
    plt.ylabel('Roots (Real part)')

    for alpha in alpha_values:
        result = np.roots([-1, 0, alpha])
        real_roots = [root.real for root in result if np.isreal(root)]

        if real_roots:
            plt.scatter(alpha, 0, s=1, marker='.', c='blue')
            plt.scatter(alpha, real_roots[0], s=1, marker='.', c='black')

            if len(real_roots) > 1:
                plt.scatter(alpha, real_roots[1], s=1, marker='.', c='black')
        else:
            plt.scatter(alpha, 0, s=1, marker='.', c='red')

    plt.xlim(alpha_initial, alpha_end)
    
    blue_legend = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=5, label='Roots Present')
    red_legend = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=5, label='No Roots')

    plt.legend(handles=[blue_legend, red_legend])
    plt.show()

def plot_bifurcation_diagram_equation7(alpha_initial: float=0, alpha_end: float=10, num_of_steps: int=1001):
    alpha_values = np.linspace(alpha_initial, alpha_end, num_of_steps)

    plt.figure(figsize=(5, 5))
    plt.title('Function 2 Bifurcation Diagram')
    plt.xlabel('Alpha')
    plt.ylabel('Roots (Real part)')

    for alpha in alpha_values:
        # Modify this part for the second function
        result = np.roots([-2, 0, (alpha -3)])  # Modify the coefficients for the second function
        real_roots = [root.real for root in result if np.isreal(root)]

        if real_roots:
            plt.scatter(alpha, 0, s=1, marker='.', c='blue')
            plt.scatter(alpha, real_roots[0], s=1, marker='.', c='black')

            if len(real_roots) > 1:
                plt.scatter(alpha, real_roots[1], s=1, marker='.', c='black')
        else:
            plt.scatter(alpha, 0, s=1, marker='.', c='red')

    plt.xlim(alpha_initial, alpha_end)

    blue_legend = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=5, label='Roots Present')
    red_legend = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=5, label='No Roots')

    plt.legend(handles=[blue_legend, red_legend])

    plt.show()
