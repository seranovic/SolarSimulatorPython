import numpy as np
from collections import Counter

def generate_star_system(
    n_planetoids=20,
    n_stars=2,
    # If `star_mass` is None, sample masses uniformly in [star_mass_min, star_mass_max].
    star_mass=None,               # None | float | list of floats (Msun)
    star_mass_min=0.1,            # Msun
    star_mass_max=2.0,            # Msun
    # Binary separation bounds (AU)
    star_sep_min=100.0,           # AU
    star_sep_max=1000.0,          # AU
    # Planetoid radial ranges relative to their host star (AU)
    planetoid_r_min=5.0,
    planetoid_r_max=6.0,
    disk_thickness=0.05,          # AU
    dv_fraction=0.05,             # fraction of circular speed used to perturb planetoid velocities
    planetoid_allocation=None,    # None => mass-weighted default; or list of fractions length n_stars
    random_seed=None,
    verbose=True                  # If True print intermediate diagnostics
):
    """
    Generate a multi-star + planetoid disk initial condition and optionally print diagnostics.

    Returns
    -------
    bodies : list of dict
        Each dict contains:
        - 'm' : mass (Msun)
        - 'r' : np.array([x, y, z]) position (AU)
        - 'v' : np.array([vx, vy, vz]) velocity (AU/yr)
    """

    # -------------------------
    # Randomness control
    # -------------------------
    # If random_seed is None, NumPy RNG is not reseeded and draws will be different each run.
    if random_seed is not None:
        np.random.seed(random_seed)

    # -------------------------
    # Physical constant (AU^3 / (Msun * yr^2))
    # -------------------------
    G = 4 * np.pi**2

    # -------------------------
    # Determine star masses
    # -------------------------
    if star_mass is None:
        if star_mass_min > star_mass_max:
            raise ValueError("star_mass_min must be <= star_mass_max")
        # Uniform sampling in linear mass between bounds
        star_masses = list(np.random.uniform(star_mass_min, star_mass_max, size=n_stars))
    elif np.isscalar(star_mass):
        star_masses = [float(star_mass)] * n_stars
    else:
        star_masses = list(star_mass)
        if len(star_masses) != n_stars:
            raise ValueError("Length of star_mass list must equal n_stars")

    total_stellar_mass = sum(star_masses)

    # -------------------------
    # Compute star positions and velocities
    # -------------------------
    star_positions = []
    star_velocities = []

    if n_stars == 2:
        # Binary: randomized separation and orientation
        m1, m2 = star_masses
        a = float(np.random.uniform(star_sep_min, star_sep_max))  # separation in AU

        if a <= 0.0:
            pos1 = pos2 = np.zeros(3)
            v1 = v2 = np.zeros(3)
        else:
            # Distances from barycenter so barycenter is at origin:
            r1 = a * m2 / (m1 + m2)
            r2 = a * m1 / (m1 + m2)

            # Random orientation in xy-plane
            theta = np.random.uniform(0.0, 2.0 * np.pi)
            ux, uy = np.cos(theta), np.sin(theta)
            # Perpendicular unit vector for tangential direction
            vx_perp, vy_perp = -uy, ux

            # Place stars at ±r_i along the chosen direction (barycenter at origin)
            pos1 = np.array([-r1 * ux, -r1 * uy, 0.0])
            pos2 = np.array([ r2 * ux,  r2 * uy, 0.0])

            # Angular frequency for circular binary: omega = sqrt(G*(m1+m2)/a^3)
            omega = np.sqrt(G * (m1 + m2) / a**3)

            # Tangential speeds v_i = omega * r_i
            v1_mag = omega * r1
            v2_mag = omega * r2

            # Assign tangential velocities perpendicular to separation vector.
            v1 = np.array([ v1_mag * vx_perp,  v1_mag * vy_perp, 0.0])
            v2 = np.array([-v2_mag * vx_perp, -v2_mag * vy_perp, 0.0])

        star_positions = [pos1, pos2]
        star_velocities = [v1, v2]

    else:
        # General n_stars: place evenly on a circle of radius star_sep_min
        radius = float(star_sep_min if star_sep_min > 0 else 1.0)
        for i in range(n_stars):
            angle = 2.0 * np.pi * i / n_stars
            pos = np.array([radius * np.cos(angle), radius * np.sin(angle), 0.0])
            star_positions.append(pos)
        for pos in star_positions:
            r = np.linalg.norm(pos)
            if r > 0:
                v_circ = np.sqrt(G * total_stellar_mass / r)
                ang = np.arctan2(pos[1], pos[0])
                v = np.array([-v_circ * np.sin(ang), v_circ * np.cos(ang), 0.0])
            else:
                v = np.zeros(3)
            star_velocities.append(v)
        # Remove net linear momentum so COM is stationary
        total_momentum = sum(m * v for m, v in zip(star_masses, star_velocities))
        v_com = total_momentum / total_stellar_mass
        star_velocities = [v - v_com for v in star_velocities]

    # -------------------------
    # Build bodies list with stars first
    # -------------------------
    bodies = []
    for m, pos, vel in zip(star_masses, star_positions, star_velocities):
        bodies.append({'m': m, 'r': pos, 'v': vel})

    # -------------------------
    # Planetoid allocation probabilities (mass-weighted default)
    # -------------------------
    if planetoid_allocation is None:
        probs = np.array(star_masses, dtype=float) / total_stellar_mass
    else:
        probs = np.array(planetoid_allocation, dtype=float)
        if probs.shape[0] != n_stars:
            raise ValueError("planetoid_allocation must have length n_stars")
        s = probs.sum()
        if s <= 0:
            raise ValueError("planetoid_allocation must sum to a positive number")
        probs = probs / s

    # Normalize per-star radial ranges (allow scalar or list)
    def _normalize_radial_param(param, name):
        if np.isscalar(param):
            return [float(param)] * n_stars
        else:
            arr = list(param)
            if len(arr) != n_stars:
                raise ValueError(f"{name} must be scalar or list of length n_stars")
            return [float(x) for x in arr]

    r_mins = _normalize_radial_param(planetoid_r_min, "planetoid_r_min")
    r_maxs = _normalize_radial_param(planetoid_r_max, "planetoid_r_max")

    # -------------------------
    # Generate planetoids (S-type: orbiting individual host stars)
    # -------------------------
    # Keep track of which host each planetoid was assigned to for diagnostics
    host_indices = []
    for _ in range(n_planetoids):
        host_idx = np.random.choice(np.arange(n_stars), p=probs)
        host_indices.append(host_idx)

        r_rel = np.random.uniform(r_mins[host_idx], r_maxs[host_idx])
        theta = np.random.uniform(0.0, 2.0 * np.pi)

        x_rel = r_rel * np.cos(theta)
        y_rel = r_rel * np.sin(theta)
        z_rel = np.random.normal(0.0, disk_thickness)

        host_pos = star_positions[host_idx]
        pos = host_pos + np.array([x_rel, y_rel, z_rel])

        m_host = star_masses[host_idx]
        v_circ_rel = 0.0 if r_rel <= 0 else np.sqrt(G * m_host / r_rel)

        dv = np.random.uniform(-dv_fraction, dv_fraction) * v_circ_rel
        v_mag_rel = v_circ_rel + dv

        vx_rel = -v_mag_rel * np.sin(theta)
        vy_rel =  v_mag_rel * np.cos(theta)
        vz_rel = np.random.normal(0.0, 0.01 * v_mag_rel)

        vel_rel = np.array([vx_rel, vy_rel, vz_rel])
        host_vel = star_velocities[host_idx]
        vel = host_vel + vel_rel

        planetoid_mass = np.random.uniform(1e-12, 1e-9)

        bodies.append({'m': planetoid_mass, 'r': pos, 'v': vel})

    # -------------------------
    # Diagnostics and verbose printing
    # -------------------------
    if verbose:
        # Star-level diagnostics
        print("\n=== STAR SYSTEM DIAGNOSTICS ===")
        for i, (m, pos, vel) in enumerate(zip(star_masses, star_positions, star_velocities)):
            print(f"Star {i}: mass = {m:.6f} Msun")
            print(f"  position (AU) = [{pos[0]:10.6f}, {pos[1]:10.6f}, {pos[2]:10.6f}]")
            print(f"  velocity (AU/yr) = [{vel[0]:10.6f}, {vel[1]:10.6f}, {vel[2]:10.6f}]")
        # Barycenter check, everything should be 0,0,0 with of course the posibility of there being round errors that are 
        # extremely tiny and non consequencial for the code.
        total_momentum = sum(m * v for m, v in zip(star_masses, star_velocities))
        total_mass = total_stellar_mass
        com_pos = sum(m * p for m, p in zip(star_masses, star_positions)) / total_mass
        com_vel = total_momentum / total_mass
        print(f"\nBarycenter position (mass-weighted) = [{com_pos[0]:10.6e}, {com_pos[1]:10.6e}, {com_pos[2]:10.6e}] AU")
        print(f"Barycenter velocity (mass-weighted) = [{com_vel[0]:10.6e}, {com_vel[1]:10.6e}, {com_vel[2]:10.6e}] AU/yr")

        # Binary separation diagnostic (if binary)
        if n_stars == 2:
            sep_vec = star_positions[1] - star_positions[0]
            sep = np.linalg.norm(sep_vec)
            print(f"\nBinary separation (distance between stars) = {sep:.6f} AU")

        # Planetoid allocation summary
        counts = Counter(host_indices)
        print("\nPlanetoid allocation (counts per star):")
        for i in range(n_stars):
            print(f"  Star {i}: {counts.get(i,0)} planetoids  (probability used = {probs[i]:.3f})")

        print("=== END DIAGNOSTICS ===\n")

    return bodies


# -------------------------
# Run generator to produce new values each execution, here is where the values should be changed if we want to 
# play around with the final numbers that are given. 
# -------------------------
if __name__ == "__main__":
    # Do not set random_seed here if you want different results each run.
    bodies = generate_star_system(
        n_planetoids=50,
        n_stars=2,
        star_mass=None,            # sample masses between star_mass_min and star_mass_max
        star_mass_min=0.1,
        star_mass_max=2.0,
        star_sep_min=100.0,
        star_sep_max=1000.0,
        planetoid_r_min=[5.0, 5.0],
        planetoid_r_max=[50.0, 50.0],
        random_seed=None,          # None => new random draws each run
        verbose=True
    )

    # Print a compact table of all bodies (stars first)
    print("m\t\t r(x,y,z)\t\t\t v(x,y,z)")
    print("-" * 80)
    for body in bodies:
        m = body["m"]
        x, y, z = body["r"]
        vx, vy, vz = body["v"]
        print(f"{m:1.2e}\t[{x:9.3f}, {y:9.3f}, {z:7.3f}]\t[{vx:9.3f}, {vy:9.3f}, {vz:7.3f}]")
