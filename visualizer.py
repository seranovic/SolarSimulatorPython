import matplotlib.pyplot as plt
import numpy as np



def display(data, is3d):

    plt.figure()
    if is3d:
        plt.axes(projection='3d')
    for idx in range(data.shape[1]):
        plt.plot(data[:, idx, 0], data[:, idx, 1], data[:, idx, 2], 'o-')
    plt.show()