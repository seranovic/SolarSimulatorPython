
def LeapFrog(forces, positions, velocities, masses, time_step,):
       x = positions[:, 0]
       y = positions[:, 1]
       forces = get_forces(positions, masses)
       accelerations = forces / masses
       velocities = velocities + accelerations * time_step
       positions = positions + velocities * time_step
       time = time + time_step



