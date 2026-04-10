import datetime

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


positions = np.array([
    [4.2e10, 0, 0],
    [-5.4e10, 0, 0],
    [0,0,0],
    [ 8.12e10, -1.44e10,  3.95e10],
    [-6.33e10,  4.29e10, -2.11e10],
    [ 3.55e10,  7.88e10,  1.01e10],
    [-9.44e10, -2.77e10,  4.88e10],
    [ 1.22e10, -8.11e10, -3.77e10],
    [ 6.90e10,  2.18e10, -4.20e10],
    [ 4.33e10, -9.80e10,  2.90e10],
    [-7.01e10,  1.99e10,  5.44e10],
    [ 2.19e10,  3.44e10, -9.33e10],
    [-5.55e10, -6.90e10,  1.22e10],
    [ 9.55e10, -4.77e10, -2.33e10],
    [-8.22e10,  6.21e10,  3.01e10],
    [ 1.55e10, -2.33e10,  8.77e10],
    [-2.11e10,  9.11e10, -4.55e10],
    [ 7.99e10,  5.22e10, -1.90e10],
    [-3.44e10, -8.77e10,  2.55e10],
    [ 6.44e10, -7.90e10,  3.90e10],
    [-1.77e10,  4.55e10, -8.12e10],
    [ 5.01e10, -3.44e10,  7.33e10],
    [-9.22e10,  2.77e10, -3.44e10],
    [ 8.55e10,  1.11e10, -6.44e10],
    [-4.90e10, -5.55e10,  7.88e10],
    [ 3.11e10,  9.44e10, -1.44e10],
    [-7.44e10,  3.33e10,  6.22e10],
    [ 2.90e10, -9.88e10,  4.11e10],
    [-6.55e10,  8.55e10, -2.88e10],
    [ 1.44e10,  5.55e10, -7.44e10],
    [ 5.88e10, -1.22e10,  9.22e10],
    [-3.22e10, -7.11e10,  4.55e10],
    [ 7.44e10,  4.99e10, -3.11e10],
    [-1.90e10,  8.22e10,  5.55e10],
    [ 9.11e10, -2.10e10, -4.11e10],
    [-8.77e10,  5.44e10,  2.88e10],
    [ 4.55e10, -6.22e10,  8.55e10],
    [-5.33e10,  1.44e10, -9.11e10],
    [ 6.99e10, -3.88e10,  1.77e10],
    [-2.44e10,  7.99e10, -5.22e10],
    [ 8.77e10, -5.90e10,  3.44e10],
    [-9.33e10,  4.22e10, -1.22e10],
    [ 2.22e10, -8.33e10,  6.11e10],
    [-7.90e10,  6.77e10, -3.88e10],
    [ 5.10e10,  2.77e10, -8.55e10],
    [-4.22e10, -9.55e10,  3.90e10],
    [ 9.44e10, -1.11e10,  5.77e10],
    [-6.99e10,  3.88e10, -7.33e10],
    [ 3.44e10, -5.88e10,  9.10e10],
    [-8.55e10,  2.22e10, -4.90e10],
    [ 1.77e10,  8.11e10,  4.22e10],
    [-5.77e10, -7.22e10, -1.99e10]

    ])

velocities = np.array([
    [+10e3, 19.5e3, -10e3],
    [-10e3, -25.2e3, +10e3],
    [0,0,0],
    [3200, -5100, 2600],
    [-4100, 3800, -2700],
    [5600, 1400, -3300],
    [-2900, -6200, 4100],
    [1200, 4800, -5200],
    [5100, -2100, 3600],
    [-4300, 5900, 1900],
    [2300, -1200, -4100],
    [3100, 5200, 1400],
    [-2400, -3100, 6200],
    [5800, -3300, -1500],
    [-5200, 4900, 2200],
    [1700, -2600, 5700],
    [-3100, 4400, -3900],
    [4200, 3500, -2300],
    [-2900, -5200, 3100],
    [4500, -4600, 1800],
    [-1600, 3100, -5300],
    [3700, -2400, 4800],
    [-5400, 1700, -3200],
    [6100, 1900, -4100],
    [-3900, -4700, 3300],
    [2600, 5800, -1400],
    [-4800, 3600, 2800],
    [2500, -6100, 3000],
    [-3300, 5400, -2100],
    [1500, 4200, -4600],
    [4800, -1100, 5200],
    [-2900, -4300, 3500],
    [5400, 3200, -2100],
    [-1800, 5100, 2600],
    [6000, -2200, -3000],
    [-5100, 3900, 2100],
    [3200, -3500, 5100],
    [-3900, 1400, -5800],
    [4500, -3100, 1500],
    [-2000, 4800, -3600],
    [5800, -3700, 3300],
    [-6100, 2800, -1100],
    [2100, -4600, 4100],
    [-5300, 4200, -2600],
    [3500, 2200, -5100],
    [-3300, -5900, 2800],
    [6200, -900, 4100],
    [-4700, 3200, -4900],
    [3100, -4200, 5500],
    [-5200, 1600, -3500],
    [1400, 4600, 2500],
    [-3600, -4900, -1500]
])

masses = np.array([
    2.19e30,
    1.69e30,
    1e10,
    3.2e20, 8.1e18, 2.5e19, 7.7e20, 4.1e17,
    6.0e19, 1.8e20, 9.4e18, 2.9e17, 3.3e19,
    1.7e21, 6.4e20, 2.2e18, 8.8e19, 1.1e20,
    2.7e19, 4.8e20, 9.9e18, 3.5e20, 7.1e19,
    1.9e21, 5.0e20, 3.3e17, 6.8e19, 4.2e20,
    1.5e20, 7.7e18, 2.1e21, 8.2e19, 3.9e20,
    9.9e18, 6.2e20, 3.8e19, 5.9e20, 2.2e20,
    8.7e17, 4.5e20, 1.8e21, 2.7e19, 5.5e20,
    3.3e18, 6.5e20, 9.9e19, 2.8e20, 4.4e19,
    1.2e21, 3.7e20, 6.6e18, 2.1e20

])



def run(pos, vel, mass, dt, steps, innersteps, force_func):
    '''Runs the simulation
    Args:
        pos: position (m) given by numpy array
        vel: velocity (ms⁻¹) given by numpy array
        mass: mass (kg) given by numpy array
        dt: time step (s)
        steps: how often to store position data
        innersteps: step * innersteps equals the total number of steps the simulation is run for
        force_func: function to calculate forces, needs to return pos, vel'''
    start = datetime.datetime.now()
    print(f'Start time of simulation: {start}')
    n, d = pos.shape
    pos_t = np.zeros((steps,n,d))
    u_t = np.zeros(steps)
    k_t = np.zeros(steps)
    p_t = np.zeros(steps)


    for step in range(steps):
        for innerstep in range(innersteps):
            forces = force_func(pos, mass)
            pos, vel = integrators.LeapFrog(forces, pos, vel, mass, dt)

        pos_t[step] = pos
        k_t[step] = tests.kinetic_energy_calc(vel, mass)
        u_t[step] = tests.potential_energy_calc(pos, mass)
        p_t[step] = tests.momentum_calc(vel,mass)

    data = {'Position': pos_t,
    'Kinetic Energy': k_t,
    'Potential Energy': u_t}
    end = datetime.datetime.now()
    print(f'End time of simulation: {end}')
    print(f'Total time of simulation: {end - start}')
    return data

if __name__ == "__main__":
    time_step = 3*60*60 # 0.125 day
    data = run(positions, velocities, masses, time_step, 1000, 1000, interactions.get_forces)

    visualizer.display(data['Position'], True)
    visualizer.display_energy(data['Kinetic Energy'], data['Potential Energy'])
    #visualizer.display_momentum(data[''])

