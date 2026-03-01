import interactions
import integrators
import numpy as np
from scipy import constants
import matplotlib.pyplot as plt

positions = np.array([
    [0,0,0],
    [0,0, constants.astronomical_unit]
])

velocities = np.array([
    [0, 0,0],
    [30e3,0,1e3]
])

masses = np.array([
    [2e30], # sun
    [6e24]
])

if __name__ == "__main__":
    time_step = 60*60*24
    time = 0
    plt.figure()
    plt.axes(projection='3d')

    while(time < 4*60*60*24*365):
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        forces = interactions.get_forces(positions, masses)
        accelerations = forces / masses
        velocities = velocities + accelerations * time_step
        positions = positions + velocities * time_step
        time = time + time_step
        print(time, x, y, z)
        plt.plot(x,y,z, 'o')

plt.show()

#find way to separate the integrator for easy hot-swap
#find better way to display 3d graph.
#find way to store data.
