'''
This is where the interactions module lives e.g. the F in F = m*a
Currently this is an O(N^2) approach.
'''

import numpy as np
import scipy.constants as constants

def get_forces_numpy(pos, mass):
    forces = np.zeros_like(pos)
    for idx in range(len(forces)):
        vectors = pos - pos[idx]
        distances_row = np.sum(vectors**2, axis=1)**0.5
        distances = distances_row[:, np.newaxis] ## currently only works on 2d, might be fun to rewrite later.
        distances[idx] = np.inf
        prefactors = constants.gravitational_constant*mass*mass[idx]/distances**3
        pair_force = prefactors*vectors
        total_force = np.sum(pair_force, axis=0)
        forces[idx] = total_force
    return forces

def get_forces(pos, mass):
    forces = np.zeros_like(pos)

    for i in range(len(forces)):
        for j in range(i+1,len(forces)):
            r_vector = pos[j] - pos[i]
            r_magnitude = np.linalg.norm(r_vector)
            if r_magnitude == 0:
                continue
            prefactors = constants.gravitational_constant*mass[i]*mass[j]/r_magnitude**3

            pair_force = prefactors*r_vector

            forces[i] += pair_force
            forces[j] -= pair_force

    return forces