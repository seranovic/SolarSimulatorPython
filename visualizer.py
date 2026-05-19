import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
import sys
import pickle


def display(data, is3d, identifier):

    plt.figure()
    if is3d:
        plt.axes(projection="3d")
    for idx in range(data.shape[1]):
        plt.plot(data[:, idx, 0], data[:, idx, 1], data[:, idx, 2], "o-")
    plt.show()
    plt.savefig(f"data/positions_{identifier}")


def display_energy(kinetic, potential, identifier):
    total = kinetic + potential
    print(f" {np.average(total)} +/- {np.std(total)}")
    plt.figure()
    plt.plot(kinetic, "-", label="Kinetic", color="red")
    plt.plot(potential, "-", label="Potential", color="blue")
    plt.plot(total, "-", label="Total Energy", color="purple")
    plt.legend()
    plt.title("Energy over time")
    plt.show()
    plt.savefig(f"data/figure_{identifier}")


def display_momentum(momentum):
    plt.figure()
    plt.plot(momentum, "-", label="Velocity")
    plt.show()

def display_energy_per(datapath):
    with open(datapath, 'rb') as f:
        data = pickle.load(f)
    positions = data['Position'][-1] # (N,3)
    velocities = data['Velocity'][-1] # (N,3)
    masses = data['Mass'] # (N)
    ke = []
    pe = []
    origo = []

    for i in range(len(masses)):
        pe_i = 0.0
        for j in range(len(masses)):
            if i == j:
                continue
            r = np.sqrt((positions[i][0] - positions[j][0])**2 + (positions[i][1] - positions[j][1])**2 + (positions[i][2] - positions[j][2])**2)
            pe_i += masses[j]/r
        pe.append(-constants.astronomical_unit*masses[i]*pe_i)


    for i in range(len(masses)):
        ke_i = 0.5*masses[i]*np.linalg.norm(velocities[i])
        ke.append(ke_i)

    print(len(ke), len(pe))

    origo = [np.linalg.norm(positions[i]) for i in range(len(masses))]

    total_E = [ke[i]+pe[i] for i in range(len(masses))]


    plt.scatter(origo[2:-1], total_E[2:-1], c=total_E[2:-1], cmap='hsv')
    plt.axhline(y=0, color='red')
    plt.xscale('log')
    plt.ylabel("Total Energy")
    plt.xlabel("Distance from origo")
    plt.savefig('meow.png')

if __name__ == "__main__":
    display_energy_per(sys.argv[1])
