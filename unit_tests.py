import numpy as np
from main import run
import interactions
import tests

def momentum_collision_elastic_test():
    positions = np.array([
        [-1, 0],
        [1, 0]
    ], dtype=np.float64)

    mass = np.array([
        1,
        1
    ], dtype=np.float64)

    velocity = np.array([
        [1, 0],
        [-1, 0]
    ], dtype=np.float64)

    radii = np.array([1, 1])
    momentum1 = tests.momentum_calc(velocity, mass)

    data = run(positions, velocity, mass, 1, 'Elastic', 0.1, 10, 10, interactions.get_forces_zeroes)

    momentum2 = tests.momentum_calc(data['Velocity'][-1,:], mass)

    if momentum1 == momentum2:
        print(velocity, data['Velocity'][-1,:])
        return True
    else:
        print(momentum1, momentum2)
        print(velocity, data['Velocity'][-1,:])
        return False

def momentum_collision_inelastic_test():
    positions = np.array([
        [-1, 0],
        [1, 0]
    ], dtype=np.float64)

    mass = np.array([
        1,
        1
    ], dtype=np.float64)

    velocity = np.array([
        [1, 0],
        [-1, 0]
    ], dtype=np.float64)

    radii = np.array([1, 1])
    momentum1 = tests.momentum_calc(velocity, mass)

    data = run(positions, velocity, mass, 1, 'Inelastic', 0.1, 10, 10, interactions.get_forces_zeroes)

    momentum2 = tests.momentum_calc(data['Velocity'][-1,:], mass)

    if momentum1 == momentum2:
        print(momentum1, momentum2)
        print(velocity, data['Velocity'][-1,:])
        return True
    else:
        print(momentum1, momentum2)
        print(velocity, data['Velocity'][-1,:])
        return False



if __name__ == '__main__':
    print(f'MOMENTUM ELASTIC TEST: {momentum_collision_elastic_test()}')
    print(f'MOMENTUM INELASTIC TEST: {momentum_collision_inelastic_test()}')



