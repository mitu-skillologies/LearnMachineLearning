import SwarmPackagePy as sw
from SwarmPackagePy import testFunctions as tf
from SwarmPackagePy import animation, animation3D
import numpy as np
sw._version_

swarm_strength = np.random.random_integers(25, 55)
print("Swarm strength =", swarm_strength)

firefly = sw.fa(n=swarm_strength,function=tf.sum_squares_function, lb=-10, ub=10, dimension=2, psi=2, iteration=50)

animation(firefly.get_agents(), tf.sum_squares_function, -10, 10)
animation3D(firefly.get_agents(), tf.sum_squares_function, -10, 10)