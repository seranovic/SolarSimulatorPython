from numba import njit
import numpy as np

def LeapFrog(forces, pos, vel, mass, dt):
       mass = np.reshape(mass, (52,1))
       accelerations = forces / mass
       vel = vel + accelerations * dt
       pos = pos + vel * dt

       return pos, vel

