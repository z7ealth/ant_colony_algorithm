import numpy as np
from ant_colony import AntColony

# Define constants for the AntColony parameters

# The distances matrix represents the distance between nodes in the graph.
# Diagonal elements are assumed to be np.inf, indicating unreachable connections.
DISTANCES = np.array([[np.inf, 3, 4, 5, 2],
                      [3, np.inf, 6, 2, 5],
                      [4, 6, np.inf, 7, 3],
                      [5, 2, 7, np.inf, 1],
                      [2, 5, 3, 1, np.inf]])

# Number of ants running per iteration.
N_ANTS = 1

# Number of best ants that deposit pheromone.
N_BEST = 1

# Number of iterations the ant colony algorithm will run.
N_ITERATIONS = 500

# Rate at which pheromone decays. The pheromone value is multiplied by DECAY,
# leading to slower decay for higher values and faster decay for lower values.
DECAY = 0.95

# Exponent on pheromone, giving pheromone more weight.
ALPHA = 1

# Exponent on distance, giving distance more weight.
BETA = 1

if __name__ == "__main__":
    ant_colony = AntColony(DISTANCES, n_ants=N_ANTS, n_best=N_BEST,
                           n_iterations=N_ITERATIONS, decay=DECAY, alpha=ALPHA, beta=BETA)
    shortest_path = ant_colony.run()

    print("\n")
    print("Shortest path: {}".format(shortest_path))
