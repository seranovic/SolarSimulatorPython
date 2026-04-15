import numpy as np
import scipy.constants as constants

def kinetic_energy_calc(vel, mass):
    K = 0
    n = len(mass)
    for i in range(n):
        K += 1 / 2 * mass[i] * np.linalg.norm(vel[i]) ** 2
    return K

def potential_energy_calc(pos, mass):
        U = 0
        n = len(mass)
        for i in range(n):
            for j in range(i + 1, n):
                r = np.linalg.norm(pos[i, :] - pos[j, :])
                U += -constants.gravitational_constant * mass[i] * mass[j] / r

        return U

def momentum_calc(vel, mass):
    P = 0
    for i in range(len(mass)):
        P += mass[i]*np.linalg.norm(vel[i])
        return P
