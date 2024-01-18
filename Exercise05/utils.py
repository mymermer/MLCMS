# Import necessary libraries
import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Function to perform least squares minimization
def least_squares_minimization(x,y,cond):
    # x: input data
    # y: output data
    # cond: condition number for the least squares problem
    return lstsq(x,y,cond=cond)[0]

# Radial basis function definition
def radial_basis_function(data,l,eps):
    # data: input data
    # l: number of radial basis functions
    # eps: shape parameter for the radial basis function
    phi = []
    np.random.seed(42)
    x_l = np.random.choice(len(data), l, replace=False)
    for i in range(l):
        normalized_values = data[x_l[i]] - data
        phi_l = np.exp(-normalized_values**2/eps**2)
        phi.append(phi_l)
    return np.array(phi)

# Function to estimate vector
def estimate_vector(x_0,x_1,delta_t):
    # x_0: initial point
    # x_1: final point
    # delta_t: time difference
    return (x_1-x_0)/delta_t

# Linear system definition
def linear_system( t, x, A):
    # t: time
    # x: state vector
    # A: system matrix
    return A @ x

# Function to generate trajectories
def generate_trajectories(x_0, t_end, A):
    # x_0: initial state
    # t_end: end time
    # A: system matrix
    t = np.linspace(0, t_end, 100)
    solution = np.zeros_like(x_0)
    for i in range(len(x_0)):
        sol = solve_ivp(linear_system, y0=x_0[i], t_span=(0, t_end), args=(A,), t_eval=t)
        solution[i] = sol.y[:, -1]
    return solution

# Function to plot data
def plot_data(data_original,approximated_data, t_end):
    # data_original: original data
    # approximated_data: data approximated by the model
    # t_end: end time
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(approximated_data[:,0],approximated_data[:,1],label="x1_hat")
    plt.scatter(data_original[:,0],data_original[:,1],label="x1")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Plot for t_end={t_end}")
    plt.legend()
    plt.show()

# Function to plot trajectory
def plot_trajectory(initial_point,t_end,A):
    # initial_point: initial state
    # t_end: end time
    # A: system matrix
    t = np.linspace(0, t_end, 100)
    sol = solve_ivp(linear_system, y0=initial_point, t_span=(0, t_end), args=(A,), t_eval=t)
    sol_x, sol_y = sol.y
    x_values, y_values = np.meshgrid(np.linspace(-10,10,20), np.linspace(-10,10,20))
    xy_vals = np.vstack([x_values.ravel(), y_values.ravel()])
    u,v = linear_system(0,xy_vals,A)
    u = u.reshape(x_values.shape)
    v = v.reshape(y_values.shape)

    fig,ax = plt.subplots(figsize=(10,10))
    ax.streamplot(x_values, y_values, u, v, color='gray', density=1, linewidth=1, arrowstyle='->')

    ax.plot(sol_x, sol_y, label='Trajectory', color='blue')

    ax.set_title("Trajectory and Vector Field")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    plt.show()
