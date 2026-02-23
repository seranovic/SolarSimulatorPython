'''
This is where the interactions module lives e.g. the F in F = m*a
Currently this is an O(N^2) approach.
'''

import numpy as np
import scipy.constants as constants

def get_forces(pos, mass):
    forces = np.zeros_like(pos)
    for idx in range(len(forces)):
        vectors = pos - pos[idx]
        distances_row = np.sum(vectors**2, axis=1)**0.5
        distances = distances_row[:, np.newaxis]
        distances[idx] = np.inf
        prefactors = constants.gravitational_constant*mass*mass[idx]/distances**3
        pair_force = prefactors*vectors
        total_force = np.sum(pair_force, axis=0)
        forces[idx] = total_force
    return forces
