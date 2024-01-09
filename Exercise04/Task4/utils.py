import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

def logistic_map(x,r):
    """
    Define the logistic map function
    Parameters:
    -----------
    x
        current population
    r
        growth rate
    """
    return r*x*(1-x)


def plot_r_x(x_values, r_values, time_steps):
    """
    Plot the logistic map for different values of r and x
    Parameters:
    -----------
    x_values
        list of x values
    r_values
        list of r values
    time_steps
        number of time steps
    """
    fig, axes = plt.subplots(len(r_values), len(x_values),figsize=(25,100))
    
    for i,x_value in enumerate(x_values):
        for j,r in enumerate(r_values):
            x = x_value
            x_list = []
            for t in range(time_steps):
                x = logistic_map(x,r)
                x_list.append(x)
            axes[j,i].plot(range(time_steps),x_list)
            axes[j, i].set_title(f'x={x_value}, r={r:.1f}')
    plt.xlabel('Time Steps')
    plt.ylabel('X')
    plt.suptitle('Logistic Map for Different Values of r and x', fontsize=20)  
    plt.subplots_adjust(top=0.97)
    plt.show()


def plot_bifurcation_diagram(x_value, r_values, time_steps):
    """
    Plot the bifurcation diagram of the logistic map
    Parameters:
    -----------
    x_value
        population
    r_values
        list of r values (growth rates)
    time_steps
        number of time steps
    """
    x_values = []
    r_plot_values = []
    for r in r_values:
        x = x_value
        for _ in range(time_steps):
            x = logistic_map(x,r)
        for _ in range(20):
            x = logistic_map(x,r)
            x_values.append(x)
            r_plot_values.append(r)

    plt.figure(figsize=(12,8))
    plt.scatter(r_plot_values, x_values, s=0.1,c ='black')
    plt.title('Bifurcation Diagram of the Logistic Map (r from 0 to 2)')
    plt.xlabel('r')
    plt.ylabel('x (steady state)')
    plt.show()

def lorenz_system(t, state, sigma, rho, beta):
    """
    Define the Lorenz system
    Parameters:
    -----------
    state
        current state
    sigma
        Prandtl number
    rho
        Rayleigh number
    beta
        geometric parameter
    """
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def plot_lorenz_attractor(solution):
    """
    Plot the Lorenz attractor
    Parameters:
    -----------
    solution
        solution of the Lorenz system
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(solution.y[0], solution.y[1], solution.y[2])
    ax.scatter(solution.y[0, 0], solution.y[1, 0], solution.y[2, 0], color='green', label='Start point')
    ax.scatter(solution.y[0, -1], solution.y[1, -1], solution.y[2, -1], color='red', label='End point')
    ax.set_title("Lorenz Attractor")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()