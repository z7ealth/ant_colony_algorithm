import numpy as np

from ant_colony import AntColony

if __name__ == "__main__":

    distances = np.array([[np.inf, 3, 4, 5, 2],
                          [3, np.inf, 6, 2, 5],
                          [4, 6, np.inf, 7, 3],
                          [5, 2, 7, np.inf, 1],
                          [2, 5, 3, 1, np.inf]])

    ant_colony = AntColony(distances, n_ants=1, n_best=1, n_iterations=500, decay=0.95, alpha=1, beta=1)
    shortest_path = ant_colony.run()
    print("\n")
    print("Shortest path: {}".format(shortest_path))
