import numpy as np


class AntColony(object):
    def __init__(self, distances, n_ants, n_best, n_iterations, decay, alpha=1, beta=1):
        """
        Initialize the AntColony with the given parameters.

        Args:
            distances (2D numpy.array): Square matrix of distances. Diagonal is assumed to be np.inf.
            n_ants (int): Number of ants running per iteration.
            n_best (int): Number of best ants who deposit pheromone.
            n_iterations (int): Number of iterations.
            decay (float): Rate at which pheromone decays. The pheromone value is multiplied by decay,
                           so 0.95 will lead to slower decay, 0.5 to much faster decay.
            alpha (int or float, optional): Exponent on pheromone, higher alpha gives pheromone more weight. Default=1.
            beta (int or float, optional): Exponent on distance, higher beta gives distance more weight. Default=1.
        """
        self.distances = distances
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self):
        """
        Run the AntColony algorithm.

        Returns:
            tuple: A tuple containing the shortest path found and its total distance.
        """
        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf)
        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheromone(all_paths, self.n_best,
                                  shortest_path=shortest_path)
            shortest_path = min(all_paths, key=lambda x: x[1])
            print(shortest_path)
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
            self.pheromone *= self.decay
        return all_time_shortest_path

    def spread_pheromone(self, all_paths, n_best, shortest_path):
        """
        Spread pheromone on the edges based on the paths taken by the ants.

        Args:
            all_paths (list): A list containing all the generated paths and their respective distances.
            n_best (int): Number of best ants that deposit pheromone.
            shortest_path: The current shortest path found during the iteration.
        """
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:n_best]:
            for move in path:
                self.pheromone[move] += 1.0 / self.distances[move]

    def gen_path_dist(self, path):
        """
        Calculate the total distance of a given path.

        Args:
            path (list): A list of nodes representing the path.

        Returns:
            float: The total distance of the path.
        """
        total_dist = sum(self.distances[ele] for ele in path)
        return total_dist

    def gen_all_paths(self):
        """
        Generate all the paths taken by the ants for a single iteration.

        Returns:
            list: A list containing tuples of paths and their respective distances.
        """
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path(0)
            all_paths.append((path, self.gen_path_dist(path)))
        return all_paths

    def gen_path(self, start):
        """
        Generate a path for a single ant starting from the given node.

        Args:
            start (int): The starting node index.

        Returns:
            list: A list of tuples representing the path, where each tuple contains two nodes representing an edge in the path.
        """
        path = []
        visited = set()
        visited.add(start)
        prev = start
        for i in range(len(self.distances) - 1):
            move = self.pick_move(
                self.pheromone[prev], self.distances[prev], visited)
            path.append((prev, move))
            prev = move
            visited.add(move)
        path.append((prev, start))  # Going back to where we started
        return path

    def pick_move(self, pheromone, dist, visited):
        """
        Select the next node for an ant based on the pheromone levels, distance, and visited nodes.

        Args:
            pheromone (numpy.array): Pheromone levels on the edges from the current node.
            dist (numpy.array): Distances from the current node to all other nodes.
            visited (set): A set containing the indices of already visited nodes.

        Returns:
            int: The index of the next node to visit.
        """
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0
        row = pheromone ** self.alpha * ((1.0 / dist) ** self.beta)
        norm_row = row / row.sum()
        move = np.random.choice(self.all_inds, 1, p=norm_row)[0]
        return move
