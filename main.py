from numba.np.arrayobj import dtype_type

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
    [2.19e30],
    [1.69e30]
])



def run(pos, vel, mass, dt, steps, innersteps, force_func):
    n, d = pos.shape
    pos_t = np.zeros((steps,n,d))

    for step in range(steps):
        for innerstep in range(innersteps):
            forces = force_func(pos, mass)
            pos, vel = integrators.LeapFrog(forces, pos, vel, mass, dt)

        pos_t[step] = pos
        print(pos)

    return pos_t




if __name__ == "__main__":

    run(positions, velocities, masses, time_step, 100, 100, interactions.get_forces_numpy)

