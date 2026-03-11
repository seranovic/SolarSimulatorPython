import interactions
import integrators
import numpy as np
from scipy import constants
import matplotlib.pyplot as plt

positions = np.array([
    [4.2e10, 0, 0],
    [-5.4e10, 0, 0]
])

velocities = np.array([
    [0, 19.5e3, 0],
    [0, -25.2e3, 0]
])

masses = np.array([
    [2.19e30],  # sun
    [1.69e30]
])

if __name__ == "__main__":
    time_step = 60 * 60 * 24
    time = 0
    time_end = 5 * 60 * 60 * 24 * 365
    plt.figure()
    plt.axes(projection='3d')

    while (time < time_end):
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        forces = interactions.get_forces(positions, masses)
        accelerations = forces / masses
        velocities = velocities + accelerations * time_step
        positions = positions + velocities * time_step
        time = time + time_step
        print(f'days are now:{time / (60 * 60 * 24)}, x:{x}, y:{y}, z:{z}')
        print(f' modulo: {time % 365 * 60 * 60 * 24}')
        if (time % 365 * 60 * 60 * 24 == 0):  # plots every year.

            plt.plot(x, y, z, 'o')
plt.show()

#find way to separate the integrator for easy hot-swap
#find better way to display 3d graph.
#find way to store data.
