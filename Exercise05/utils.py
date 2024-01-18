import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def least_squares_minimization(x,y,cond):
    return lstsq(x,y,cond=cond)[0]

# radial basis function definition
def radial_basis_function(data,l,eps):
    phi = []
    np.random.seed(42)
    x_l = np.random.choice(len(data), l, replace=False)
    for i in range(l):
        normalized_values = data[x_l[i]] - data
        phi_l = np.exp(-normalized_values**2/eps**2)
        phi.append(phi_l)
    return np.array(phi)

def estimate_vector(x_0,x_1,delta_t):
    return (x_1-x_0)/delta_t

def linear_system( t, x, A):
    return A @ x

def generate_trajectories(x_0, t_end, A):
    t = np.linspace(0, t_end, 100)
    solution = np.zeros_like(x_0)
    for i in range(len(x_0)):
        sol = solve_ivp(linear_system, y0=x_0[i], t_span=(0, t_end), args=(A,), t_eval=t)
        solution[i] = sol.y[:, -1]
    return solution

def plot_data(data_original,approximated_data, t_end):
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(approximated_data[:,0],approximated_data[:,1],label="x1_hat")
    plt.scatter(data_original[:,0],data_original[:,1],label="x1")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Plot for t_end={t_end}")
    plt.legend()
    plt.show()

def plot_trajectory(initial_point,t_end,A):
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