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

def array_maker(pos, end_time, factor):
    '''Creates a numpy array that stores the positions value of each celestial object.'''
    n = int(pos.size/3) #number of objects
    #print(n)
    t = int(end_time/factor) #number of total timesteps
    #print(t)

    pos = np.zeros((t,n,3)) # first number is column number, e.g. time, second and third number give dimensonality
    # we have n objects with 3 dimensions so should be t*n*3 array.

    return pos

def run(pos, vel, masses, steps, innersteps, force.func):
    n, d = pos.positions.shape
    pos_t = np.zeroes(steps,n,d)

    for step in range(steps):
        for innerstep in range(innersteps):



if __name__ == "__main__":
    time_step = 60 * 60 * 24 # 1 day
    time = 0
    time_end = 5 * 60 * 60 * 24 * 365 # 5 years
    factor = 365*60*60*24 # every year
    plt.figure()
    plt.axes(projection='3d')

    while (time < time_end):
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        forces = interactions.get_forces(positions, masses)
        accelerations = forces / masses
        velocities = velocities + accelerations * time_step
        positions = positions + velocities * time_step
        time = time + time_step
        print(f'days are now:{time / (60 * 60 * 24)}, x:{x}, y:{y}, z:{z}')
        print(f' modulo: {time % 365 * 60 * 60 * 24}')
        print(f' positions np.array: {positions}')
        if (time % factor == 0):  # plots every year.
            plt.plot(x, y, z, 'o')
#plt.show()

print(array_maker(positions, time_step, time_end,factor))


#find way to separate the integrator for easy hot-swap
#find better way to display 3d graph.
#find way to store data.
