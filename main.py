from numba.np.arrayobj import dtype_type

import interactions
import integrators
import numpy as np
import visualizer
from scipy import constants
import tests
import matplotlib.pyplot as plt
from numba import njit
import pickle
import time
import pandas as pd


positions = np.array(
    [
        [10, 0, 0],
        [-10, 0, 0],
    ],
    dtype=np.float64,
)

velocities = np.array(
    [
        [-1, 0, 0],
        [0, 0, 0],
    ],
    dtype=np.float64,
)

masses = np.array(
    [
        2.19e30,
        2 * 2.19e30,
    ],
    dtype=np.float64,
)


def radii_maker(mass, density):  # unsure if this works rn lolz
    radii = np.zeros_like(masses)
    for i in range(len(radii)):
        radii[i] = ((3 * mass[i]) / (4 * np.pi * density)) ** 1 / 3

    return radii


def run(pos, vel, mass, radii, collision, dt, steps, innersteps, force_func):
    """Runs the simulation
    Args:
        pos: position (m) given by numpy array/
        vel: velocity (ms⁻¹) given by numpy array
        mass: mass (kg) given by numpy array
        dt: time step (s)
        steps: how often to store position data
        innersteps: step * innersteps equals the total number of steps the simulation is run for
        force_func: function to calculate forces, needs to return pos, vel"""
    start = time.perf_counter()
    n, d = pos.shape
    pos_t = np.zeros((steps, n, d))
    u_t = np.zeros(steps)
    k_t = np.zeros(steps)
    p_t = np.zeros(steps)
    vel_t = np.zeros((steps, n, d))

    for step in range(steps):
        for innerstep in range(innersteps):
            forces = force_func(pos, mass)
            if collision == True:
                vel = interactions.handle_collisions(pos, vel, mass, radius=radii)
            pos, vel = integrators.LeapFrog(forces, pos, vel, mass, dt)

        pos_t[step] = pos
        k_t[step] = tests.kinetic_energy_calc(vel, mass)
        u_t[step] = tests.potential_energy_calc(pos, mass)
        p_t[step] = tests.momentum_calc(vel, mass)
        vel_t[step] = vel

    data = {
        "Position": pos_t,
        "Kinetic Energy": k_t,
        "Potential Energy": u_t,
        "Velocity": vel_t,
    }
    end = time.perf_counter()
    print(f"Total time of simulation: {end - start} s")
    return data


if __name__ == "__main__":
    time_step = 3 * 60 * 60  # 0.125 day
    data = run(
        positions,
        velocities,
        masses,
        1,
        True,
        time_step,
        100,
        100,
        interactions.get_forces,
    )

    with open(
        "data.pkl", "wb"
    ) as f:  # this saves the data so it can be 'depickled' later.
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    visualizer.display(data["Position"], True)
    visualizer.display_energy(data["Kinetic Energy"], data["Potential Energy"])
    # visualizer.display_momentum(data[''])
