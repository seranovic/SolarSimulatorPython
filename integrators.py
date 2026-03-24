
def LeapFrog(forces, pos, vel, mass, dt):

       accelerations = forces / mass
       vel = vel + accelerations * dt
       pos = pos + vel * dt

       return pos, vel

