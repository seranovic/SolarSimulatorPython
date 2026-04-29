'''
This is where the interactions module lives e.g. the F in F = m*a
Currently this is an O(N^2) approach.
'''

import numpy as np
import scipy.constants as constants
import numba

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
@numba.njit(parallel=True)
def get_forces(pos, mass):
    forces = np.zeros_like(pos)

    for i in numba.prange(len(forces)):
        for j in range(i+1,len(forces)):
            r_vector = pos[j,:] - pos[i,:]
            r_magnitude = np.linalg.norm(r_vector)
            if r_magnitude == 0:
                continue
            prefactors = constants.gravitational_constant*mass[i]*mass[j]/r_magnitude**3
            #print(prefactors.shape, np.dtype(prefactors))
            #print(r_vector.shape, np.dtype(r_vector))

            pair_force = prefactors*r_vector

            forces[i,:] += pair_force
            forces[j,:] -= pair_force

    return forces

def get_forces_zeroes(pos, mass):
    forces = np.zeros_like(pos)
    return forces

# Elastic Collisions (3D), unsure where to go next for inelastic colissions
@numba.njit(parallel=True)
def handle_collisions(pos, vel, mass, radius):
    for i in numba.prange(len(pos)):
        for j in range(i + 1, len(pos)):
            r = pos[i, :] - pos[j, :]
            dist = np.linalg.norm(r)
            if dist < radius:
                n_hat = r / dist
                v_rel = vel[i, :] - vel[j, :]
                vn = np.dot(v_rel, n_hat)

                if vn > 0:
                    continue  # bodies are separating
                m1, m2 = mass[i], mass[j]
                print(m1, m2)
                vel[i, :] -= (2 * m2 / (m1 + m2)) * vn * n_hat
                vel[j, :] += (2 * m1 / (m1 + m2)) * vn * n_hat

    return vel