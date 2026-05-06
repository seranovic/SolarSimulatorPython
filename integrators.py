from numba import njit
import numpy as np

def LeapFrog(forces, pos, vel, mass, dt):
       """:arg forces: forces
       :arg pos: position
       :arg vel: velocity
       :arg mass: mass
       :arg dt: time
       :return: pos, vel
       """
       mass = np.reshape(mass, (len(mass),1))
       accelerations = forces / mass
       vel = vel + accelerations * dt
       pos = pos + vel * dt

       return pos, vel

