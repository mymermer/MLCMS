import matplotlib.pyplot as plt

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
    ax.plot(solution.y[0], solution.y[1], solution.y[2],lw=0.5)
    ax.scatter(solution.y[0, 0], solution.y[1, 0], solution.y[2, 0], color='green', label='Start point')
    ax.scatter(solution.y[0, -1], solution.y[1, -1], solution.y[2, -1], color='red', label='End point')
    ax.set_title("Lorenz Attractor")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()