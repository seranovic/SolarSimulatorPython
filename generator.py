import numpy as np
import pickle
from scipy import constants

# ============================================================
# Star system generator
# ============================================================

def generate_star_system(
    n_planetoids=100,
    star_mass=2.0,          # Solar masses
    r_min=5.0,              # AU
    r_max=6.0,              # AU
    disk_thickness=0.05,    # AU (z scatter)
    dv_fraction=0.05,       # |Δv| as fraction of v_circ
    random_seed=None
):
    """
    Generate a simple star + planetoid disk initial condition.

    Returns
    -------
    bodies : list of dict
        Each dict contains:
        - 'm' : mass
        - 'r' : np.array([x, y, z])
        - 'v' : np.array([vx, vy, vz])
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    # Gravitational constant in units:
    # AU^3 / (Msun * yr^2)  which means 1 AU/yr = 4 740.57172 m / s
    G = 4 * np.pi**2

    bodies = []

    posit = []
    velo = []
    mass = []

    # --------------------------------------------------------
    # Central star
    # --------------------------------------------------------
    star = {
        'm': star_mass,
        'r': np.array([0.0, 0.0, 0.0]),
        'v': np.array([0.0, 0.0, 0.0])
    }

    posit.append(star['r'])
    velo.append(star['v'])
    mass.append(star_mass)

    bodies.append(star)

    # --------------------------------------------------------
    # Planetoids
    # --------------------------------------------------------
    for _ in range(n_planetoids):

        # Random orbital radius
        r = np.random.uniform(r_min, r_max)

        # Random orbital angle (0–2π)
        theta = np.random.uniform(0.0, 2.0 * np.pi)

        # Position (disk in xy plane)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.random.normal(0.0, disk_thickness)

        pos = np.array([x, y, z])

        # Circular velocity
        v_circ = np.sqrt(G * star_mass / r)

        # Add small velocity perturbation
        dv = np.random.uniform(-dv_fraction, dv_fraction) * v_circ
        v_mag = v_circ + dv

        # Tangential velocity
        vx = -v_mag * np.sin(theta)
        vy =  v_mag * np.cos(theta)
        vz = np.random.normal(0.0, 0.01 * v_mag) #peturbations in the z-axis to simulate how planetesimals move in 3D space

        vel = np.array([vx, vy, vz])

        planetoid = {
            'm': np.random.uniform(1e-12, 1e-9),
            'r': pos,
            'v': vel
        }

        posit.append(pos)
        velo.append(vel)
        mass.append(planetoid['m'])

        bodies.append(planetoid)


        positions = np.asarray(posit)
        velocities = np.asarray(velo)
        masses = np.asarray(mass)


    return positions, velocities, masses


# ============================================================
# MAIN (PRINT OUTPUT)
# ============================================================

if __name__ == "__main__":

    # Generate system
    pos, vel, m = generate_star_system(random_seed=42)
    pos = constants.astronomical_unit*pos
    vel = 4740.57*vel
    m = 1.98e30*m

    bodies = {'positions': pos,
            'velocities': vel,
            'mass' : m}
    with open('initial_conditions.pkl', 'wb') as f:
        pickle.dump(bodies, f)


    # Print header
#    print("m\t\t\t\t r(x, y, z)\t\t\t\t\t v(x, y, z)")
#    print("-" * 80)
#
    # Print all bodies
#    for body in bodies:
#        m = body["m"]
#        x, y, z = body["r"]
#        vx, vy, vz = body["v"]
#
#        print(
#            f"{m:1.2e}\t"
#            f"[{x:7.3f}, {y:7.3f}, {z:7.3f}]\t"
#            f"[{vx:7.3f}, {vy:7.3f}, {vz:7.3f}]"
#        )

