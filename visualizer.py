import matplotlib.pyplot as plt
import numpy as np



def display(data, is3d, identifier):

    plt.figure()
    if is3d:
        plt.axes(projection='3d')
    for idx in range(data.shape[1]):
        plt.plot(data[:, idx, 0], data[:, idx, 1], data[:, idx, 2], 'o-')
    plt.show()
    plt.savefig(f'Positions {identifier}')

def display_energy(kinetic, potential, identifier):
    total = kinetic+potential
    print(f' {np.average(total)} +/- {np.std(total)}')
    plt.figure()
    plt.plot(kinetic, '-', label='Kinetic', color='red')
    plt.plot(potential, '-', label='Potential', color='blue')
    plt.plot(total, '-', label='Total Energy', color='purple')
    plt.legend()
    plt.title('Energy over time')
    plt.show()
    plt.savefig(f'Figure {identifier}')

def display_momentum(momentum):
    plt.figure()
    plt.plot(momentum, '-', label='Velocity')
    plt.show()
