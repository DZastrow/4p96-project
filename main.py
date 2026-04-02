import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections import namedtuple
import load_experiments as le
import os
import time

City = namedtuple("City", ["id", "x", "y"])



def load_tsp(file_path):
  cities = []
  with open(file_path, 'r') as f:
    lines = f.readlines()
  reading_coords = False
  for line in lines:
    line = line.strip() # clean whitespace
    if line == "NODE_COORD_SECTION":  # the .tsp files are all denoted such that this marks the start of the cities
      reading_coords = True
      continue
    if line == "EOF": # self explanatory
      break
    if reading_coords:
      parts = line.split()  # split each line into its components: node, x_coord, y_coord
      if len(parts) >= 3:
        cities.append(City(int(parts[0]), float(parts[1]), float(parts[2])))
  return cities

def load_tour(file_path):
  tour = []
  with open(file_path, 'r') as f:
    lines = f.readlines()
  reading = False
  for line in lines:  # pretty much the same as load_tsp
    line = line.strip()
    if line == "TOUR_SECTION":
      reading = True
      continue
    if line == "-1":
      break
    if reading:
      tour.append(int(line))
  return tour



def euclidean_distance(c1, c2):
  return math.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)

def compute_distance_matrix(cities):  # calculate the distances between each point
  n = len(cities)
  dist = np.zeros((n, n))
  for i in range(n):
    for j in range(n):
      if i != j:
        dist[i][j] = euclidean_distance(cities[i], cities[j])
  return dist



class AntColony:
  def __init__(self, dist_matrix, n_ants, n_epochs, alpha, beta, evaporation, Q):
    self.dist = dist_matrix
    self.n = len(dist_matrix)
    self.n_ants = n_ants
    self.n_epochs = n_epochs
    self.alpha = alpha  # strength of pheromone in decision-making
    self.beta = beta  # strength of heuristic in decision-making
    self.evaporation = evaporation # evap rate
    self.Q = Q  # strength of pheromone depositing
    self.pheromone = np.ones((self.n, self.n))  # initialize every path to the same pheromone level so there's nothing immediately preferred
    self.heuristic = 1/dist_matrix  # shorter distance = larger heuristic value

  def run(self):
    print("Starting ACO optimization...")
    start_time = time.time()
    best_length = float('inf')  # start with best_length set to infinite so that it can only go down
    best_path = None
    convergence = []
    for epoch in range(self.n_epochs):
      # print("starting epoch", epoch+1)
      all_paths = []  # start each epoch with a fresh set of paths to be found
      all_lengths = []
      for _ in range(self.n_ants):
        # print("  Ant", _+1, "is finding a path...")
        path = self.find_path()  # each ant looks for a path on its own
        length = self.path_length(path)
        all_paths.append(path)
        all_lengths.append(length)
        if length < best_length:  # is the path better? keep it
          best_length = length
          best_path = path
      self.update_pheromones(all_paths, all_lengths)  # have to evaporate all and re-apply on the traveled paths
      convergence.append(best_length) # for graphing purposes
      print(f"Epoch {epoch+1}: Best = {best_length:.2f}")
      ##### probably put in some auto stop when it converges
    end_time = time.time()
    elapsed_time = end_time - start_time
    return best_path, best_length, convergence, elapsed_time

  def find_path(self):
    start = random.randint(0, self.n - 1) # every ant starts at a random city for more diverse exploration
    path = [start]
    visited = set(path)
    for _ in range(self.n - 1):
      current = path[-1]
      next_city = self.select_next_city(current, visited) # pick the next point and go to it. repeat through the whole thing so we get a full path
      path.append(next_city)
      visited.add(next_city)
    return path

  def select_next_city(self, current, visited):
    cities = []
    probs = []
    for j in range(self.n):
      if j not in visited:  # we only want to visit each city once, so don't bother checking the already visited ones
        tau = self.pheromone[current][j] ** self.alpha  # alpha and beta influence the strength of the pheromone and heuristic
        eta = self.heuristic[current][j] ** self.beta
        cities.append(j)
        probs.append(tau * eta)
    probs = np.array(probs)
    probs /= probs.sum()
    return np.random.choice(cities, p=probs)  # randomly choose a city based on the probability distribution created in probs

  def path_length(self, path):
    length = 0
    for i in range(len(path)):
      length += self.dist[path[i]][path[(i + 1) % len(path)]] # basic
    return length

  def update_pheromones(self, paths, lengths):
    self.pheromone *= (1 - self.evaporation)  # evaporate everything a bit
    for path, length in zip(paths, lengths):
      for i in range(len(path)):
        a = path[i]
        b = path[(i + 1) % len(path)]
        self.pheromone[a][b] += self.Q / length # deposit pheromone based on path length; we don't really want other ants going down long paths too often
        self.pheromone[b][a] += self.Q / length



def plot_tour(cities, path, title="Tour"):
  x = [cities[i].x for i in path] + [cities[path[0]].x]
  y = [cities[i].y for i in path] + [cities[path[0]].y]
  plt.figure()
  plt.plot(x, y, marker='o')  # draw out the map of cities and the path taken
  plt.title(title)
  plt.show()

def plot_convergence(convergence):
  plt.figure()
  plt.plot(convergence)
  plt.title("Convergence (Best Path Length Over Time)")
  plt.xlabel("Epoch")
  plt.ylabel("Path Length")
  plt.show()



def run_experiment(experiments, n_ants, n_epochs, alpha, beta, evaporation):
  for tsp_file, tour_file in experiments:
    print(f"\nRunning experiment on {tsp_file} ...")
    cities = load_tsp(tsp_file) # read the .tsp file and construct the list of cities
    print(f"Loaded {len(cities)} cities.")
    dist_matrix = compute_distance_matrix(cities)
    print("Distance matrix computed.")
    aco = AntColony(dist_matrix, n_ants, n_epochs, alpha, beta, evaporation, 100)
    print("Running ACO...")
    best_path, best_length, convergence, elapsed_time = aco.run()

    print("\n##### RESULTS #####")
    print("Best path length:", best_length)
    print("Elapsed time (seconds): {:.2f}".format(elapsed_time))
    if os.path.exists(tour_file):# if we have a known optimal path in a .tour file, compare our best path to that
      optimal_tour = load_tour(tour_file)
      optimal_path = [i - 1 for i in optimal_tour]
      optimal_length = aco.path_length(optimal_path)
      print("Optimal length:", optimal_length)
      print("Error (%):", 100 * (best_length - optimal_length) / optimal_length)
    else:
      print("No optimal tour file found for comparison.")

    #plot_tour(cities, best_path, f"{tsp_file} - ACO Best Path") # graph the best found path
    #plot_convergence(convergence) # also graph the path distance over time
    
    ##### should we measure calculation time?

# tsp_file = "res/pr439.tsp"  ##### need to replace this with "go through the entire folder and do them all"
# tour_file = "res/pr439.opt.tour"

output_folder = "experiment_results"  # where we want to save the results of our experiments
# make array of string names of all the .tsp files in the folder

experiments = le.return_experiments() # get the list of experiments from load_experiments.py
le.print_experiments(experiments) # print out the experiments so we can see what we're working with

# to run single experiment:
# experiments = [(tsp_file, tour_file)]
run_experiment(experiments, 15, 5, 1.0, 5.0, 0.5)

