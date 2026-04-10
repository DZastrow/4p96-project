import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections import namedtuple
import os
import time
import seaborn as sns

City = namedtuple("City", ["id", "x", "y"])



def return_experiments(): # fetch all .tsp files
    experiments = []
    for file in os.listdir("res"):
        if file.endswith(".tsp"):
            tsp_file = os.path.join("res", file)
            tour_file = os.path.join("res", file.replace(".tsp", ".opt.tour"))
            experiments.append((tsp_file, tour_file))
    return experiments

def return_experiments_with_tours(): # fetch all .tsp files that have a corresponding .tour file; these will be the asymmetric ones
    experiments = []
    for file in os.listdir("res"):
        if file.endswith(".tsp"):
            tsp_file = os.path.join("res", file)
            tour_file = os.path.join("res", file.replace(".tsp", ".opt.tour"))
            if os.path.exists(tour_file):
                experiments.append((tsp_file, tour_file))
    return experiments

def print_experiments(experiments):
    for tsp_file, tour_file in experiments:
        print(f"TSP file: {tsp_file}, Tour file: {tour_file}")

def load_tsp(file_path): # parse a .tsp file into usable data
    cities = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    reading_coords = False
    for line in lines:
        line = line.strip() # clean whitespace
        if line == "NODE_COORD_SECTION" or line == "DISPLAY_DATA_SECTION": # the .tsp files are all denoted such that this marks the start of the cities
            reading_coords = True
            continue
        if line == "EOF": # self explanatory
            break
        if reading_coords:
            parts = line.split() # split each line into its components: node, x_coord, y_coord
            if len(parts) > 0 and not parts[0].isdigit(): # if we hit another section header, stop reading
                reading_coords = False
                continue
            if len(parts) >= 3:
                cities.append(City(int(parts[0]), float(parts[1]), float(parts[2])))
    return cities

def load_tsp_flexible(file_path): # it turned out that some .tsp files are structured as coordinates and some already come as matrices. we can't use the same operations for both so a second method has been implemented; load_tsp() stays for posterity
    with open(file_path, 'r') as f:
        lines = f.readlines()
    info = {}
    matrix_data = []
    coords = []
    reading_matrix = False
    reading_coords = False
    for line in lines:
        line = line.strip()
        if not line or line == "EOF": 
            break
        if ":" in line and not (reading_matrix or reading_coords):
            key, val = line.split(":", 1)
            info[key.strip()] = val.strip()
            continue
        if line == "EDGE_WEIGHT_SECTION":
            reading_matrix = True
            reading_coords = False
            continue
        if line in ["NODE_COORD_SECTION", "DISPLAY_DATA_SECTION"]:
            reading_coords = True
            reading_matrix = False
            continue
        parts = line.split()
        if reading_matrix:
            matrix_data.extend([float(x) for x in parts])
        elif reading_coords and len(parts) >= 3:
            coords.append(City(int(parts[0]), float(parts[1]), float(parts[2])))
    if coords:
        return "COORDS", coords
    elif matrix_data:
        n = int(info['DIMENSION'])
        dist_matrix = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist_matrix[i][j] = matrix_data[idx]
                dist_matrix[j][i] = matrix_data[idx]
                idx += 1
        return "MATRIX", dist_matrix
    return None, None

def load_tour(file_path): # parse a .tour format into usable data. basically the same as load_tsp()
    tour = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    reading = False
    for line in lines:
        line = line.strip()
        if not line: 
          continue
        if line == "TOUR_SECTION": # .tour files are also uniformly sectioned
          reading = True
          continue
        if line == "-1" or line == "EOF":
          break
        if reading:
          parts = line.split() # split the line by spaces in case there are multiple numbers, some files have it with one city per line and some have multiple
          for part in parts:
            if part == "-1": # some files put -1 at the end of the line
              return tour
            try:
              tour.append(int(part))
            except ValueError:
              continue # skip things that aren't numbers, just in case
    return tour



def euclidean_distance(c1, c2): # individual distance
    d = math.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)
    return int(d + 0.5) # TSPLIB standard: round up to the nearest integer

def compute_distance_matrix(cities): # calculate the distances between each point
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
        self.alpha = alpha # strength of pheromone in decision-making
        self.beta = beta # strength of heuristic in decision-making
        self.evaporation = evaporation # evap rate
        self.Q = Q # strength of pheromone depositing
        self.pheromone = np.ones((self.n, self.n)) # initialize every path to the same pheromone level so there's nothing immediately preferred
        self.heuristic = 1.0 / (dist_matrix + 1e-10) # shorter distance = larger heuristic value
                                # 1e-10 gets rid of potential divide by zero errors

    def run(self):
        print("Starting ACO optimization...")
        best_length = float('inf') # start with best_length set to infinite so that it can only go down
        best_path = None
        convergence = []
        patience = 200 # algorithm stops early if we haven't seen any improvement in 200 epochs
        no_improv = 0 # patience counter
        discovery_epoch = 0 # epoch at which we found the best path, for analysis purposes
        found_new_best = False # flag to track if we found a new best path in the current epoch
        start_time = time.time() # start the timer so we can keep track of how long it takes this instance to run
        for epoch in range(self.n_epochs):
            # print("starting epoch", epoch+1)
            all_paths = [] # start each epoch with a fresh set of paths to be found
            all_lengths = []
            for _ in range(self.n_ants):
                # print("  Ant", _+1, "is finding a path...")
                path = self.find_path() # each ant looks for its own path
                length = self.path_length(path)
                all_paths.append(path)
                all_lengths.append(length)
                if length < best_length: # is the path better than what we have? if so, it's our new best
                    best_length = length
                    best_path = path
                    found_new_best = True
                    discovery_epoch = epoch + 1 # record when we found this new best path
            if found_new_best:
                no_improv = 0  # reset patience counter if we found a new best path
                found_new_best = False  # reset flag for the next epoch
            else:
                no_improv += 1  # increment patience counter if we didn't find a new best path
            self.update_pheromones(all_paths, all_lengths)  # have to evaporate all and re-apply on the traveled paths
            convergence.append(best_length) # for graphing purposes
            #print(f"Epoch {epoch+1}: Best = {best_length:.2f}")
            if no_improv >= patience:
                print(f"No improvement in {patience} epochs, stopping early.")
                break
            #if (epoch + 1) % 10 == 0 or epoch == 1:
                #plot_pheromone_heatmap(self.pheromone, epoch+1, tsp_file)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return best_path, best_length, convergence, elapsed_time, discovery_epoch

    def find_path(self):
        start = random.randint(0, self.n - 1) # every ant starts at a random city for more diverse exploration
        path = [start]
        visited = set(path)
        for _ in range(self.n - 1): # must visit all cities
            current = path[-1]
            next_city = self.select_next_city(current, visited) # pick the next point and go to it. repeat through the whole thing so we get a full path
            path.append(next_city)
            visited.add(next_city)
        return path

    def select_next_city(self, current, visited):
        cities = []
        probs = []
        for j in range(self.n):
            if j not in visited: # we only want to visit each city once, so don't bother checking the already visited ones
                tau = self.pheromone[current][j] ** self.alpha # alpha and beta influence the strength of the pheromone and heuristic
                eta = self.heuristic[current][j] ** self.beta
                cities.append(j)
                probs.append(tau * eta)
        probs = np.array(probs)
        probs /= probs.sum()
        return np.random.choice(cities, p=probs) # randomly choose a city based on the probability distribution created in probs

    def path_length(self, path):
        length = 0
        for i in range(len(path)):
            length += self.dist[path[i]][path[(i + 1) % len(path)]] # pretty basic
        return length

    def update_pheromones(self, paths, lengths):
        self.pheromone *= (1 - self.evaporation) # evaporate everything based on that rate
        for path, length in zip(paths, lengths):
            for i in range(len(path)):
                a = path[i]
                b = path[(i + 1) % len(path)]
                self.pheromone[a][b] += self.Q / length # deposit pheromone based on path length; we don't really want other ants going down long paths too often
                self.pheromone[b][a] += self.Q / length



def calculate_mre(trial_results, optimal_value):
    mean_length = np.mean(trial_results)
    mre = ((mean_length - optimal_value) / optimal_value) * 100
    return mre



def plot_tour(cities, path, tsp_file): # plot a path on a set of cities. only used for found best paths here
    x = [cities[i].x for i in path] + [cities[path[0]].x]
    y = [cities[i].y for i in path] + [cities[path[0]].y]
    plt.figure()
    plt.plot(x, y, marker='o') # draw out the map of cities and the path taken
    plt.title(f"{tsp_file} best path")
    plt.show()
    plt.savefig(f"{tsp_file}_tour.png", dpi=300)

def plot_convergence(convergence, tsp_file): # path length over time
    plt.figure()
    plt.plot(convergence)
    plt.title(f"Best path length over time for {tsp_file}")
    plt.xlabel("Epoch")
    plt.ylabel("Path Length")
    plt.show()
    plt.savefig(f"{tsp_file}_convergence.png", dpi=300)

def plot_pheromone_heatmap(pheromone_matrix, iteration, instance_name): # levels of pheromone deposition over time
    plt.figure(figsize=(10,8))
    ax = sns.heatmap(pheromone_matrix, cmap="YlGnBu", robust=True)
    plt.title(f"Pheromone Heatmap - {instance_name} - Iteration {iteration}")
    plt.xlabel("City Index")
    plt.ylabel("City Index")
    plt.savefig(f"pheromone_heatmap_{instance_name}_iter{iteration}.png", dpi=300)
    print(f"Saved pheromone heatmap for {instance_name} iteration {iteration}")
    plt.close()

def plot_optimal_tour(tsp_file, tour_file): # plot the optimal path from a tour file onto its set of cities, to see how it looks
    cities = load_tsp(tsp_file)
    tour = load_tour(tour_file)
    path = [city_id - 1 for city_id in tour]
    x = [cities[i].x for i in path]
    y = [cities[i].y for i in path]
    x.append(cities[path[0]].x)
    y.append(cities[path[0]].y)
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.title(f"{tsp_file} optimal path")
    plt.xticks([])
    plt.yticks([])
    plt.show()



def run_experiment(experiments, n_ants, n_epochs, alpha, beta, evaporation, num_trials):
    all_summary_results = {} # save results to calculate averages and such later on
    for tsp_file, tour_file in experiments: # run all the experiments

        data_type, data = load_tsp_flexible(tsp_file) # pull up the data for this experiment
        if data_type == "COORDS":
            cities = data
            dist_matrix = compute_distance_matrix(cities)
        elif data_type == "MATRIX":
            cities = None # no way to plot these!
            dist_matrix = data
        else:
            print(f"Skipping {tsp_file}: Unsupported format.") # Just In Case (tm)
            continue

        trial_lengths = []
        trial_times = []
        trial_errors = []
        trial_convergences = []
        trial_discovery_epochs = []
        for i in range(num_trials): # test each experiment x number of times so we can collect averages
            print(f"\nRunning experiment on {tsp_file}, {tour_file} ...")
            print(f"Trial {i+1}/{num_trials}")
            aco = AntColony(dist_matrix, n_ants, n_epochs, alpha, beta, evaporation, 100)
            print("Running ACO...")
            best_path, best_length, convergence, elapsed_time, discovery_epoch = aco.run()
            trial_lengths.append(best_length)
            trial_times.append(elapsed_time)
            trial_convergences.append(convergence)
            trial_discovery_epochs.append(discovery_epoch)
            print("\n##### RESULTS #####")
            print("Best path length:", best_length)
            print("Elapsed time (seconds): {:.2f}".format(elapsed_time))

            if os.path.exists(tour_file): # if we have a known optimal path in a .tour file for this .tsp, compare our best path to that
                optimal_tour = load_tour(tour_file)
                optimal_path = [i - 1 for i in optimal_tour]
                optimal_length = aco.path_length(optimal_path)
                error =  100 * (best_length - optimal_length) / optimal_length
                trial_errors.append(error)
                print("Optimal length:", optimal_length)
                print("Error (%):", error)
            else:
                print("No optimal tour file found for comparison.")

        max_len = max(len(c) for c in trial_convergences) # pad the convergence arrays in case some ran for more epochs
        padded_convergences = []
        for c in trial_convergences:
            padding = [c[-1]] * (max_len - len(c))
            padded_convergences.append(c + padding)
        all_summary_results[tsp_file] = { # calculate and store summary results for this experiment
            "average_length": np.mean(trial_lengths),
            "average_time": np.mean(trial_times),
            "average_convergence": np.mean(padded_convergences, axis=0),
            "best_length": np.min(trial_lengths),
            "best_path": best_path,
            "optimal_length": optimal_length if os.path.exists(tour_file) else None,
            "discovery_epoch": np.mean(trial_discovery_epochs),
            "standard_deviation_length": np.std(trial_lengths)
        }
        tsp_file_name = tsp_file.split('/') # they're all in the format of "res/ulysses22.tsp" and that doesn't look too pretty on a graph so we'll trim them down
        tsp_file_name = tsp_file_name[1].split('.')
        if (cities != None):
            plot_tour(cities, best_path, tsp_file_name[0])  # graph the best found path
        plot_convergence(np.mean(padded_convergences, axis=0), tsp_file_name[0])  # also graph the path distance over time

    print(f"done Avg Length: {all_summary_results[tsp_file]['average_length']:.2f}")
    return all_summary_results



experiments = return_experiments_with_tours() # get the list of experiments from load_experiments.py
print_experiments(experiments) # print out the experiments so we can see what we'll be working with

#  to run a single experiment, uncomment this block and replace tsp_file, tour_file with whichever files you want to run
"""tsp_file = r"res/ulysses22.tsp"
tour_file = r"res/ulysses22.opt.tour"
experiments = [(tsp_file, tour_file)]"""

                            # experiments, n_ants, n_epochs, alpha, beta, evaporation, num_trials
all_results = run_experiment(experiments, 10, 100, 1.0, 2.0, 0.9, 10)

print(f"File Name | Average Length | Average Time (sec) | Best Length | Optimal Length | MRE (%) | StDev of Lengths")
for tsp_file, results in all_results.items():
    print(f"\nSummary for {tsp_file}:")
    print(f"Average Length: {results['average_length']:.2f}")
    print(f"Average Time: {results['average_time']:.2f} seconds")
    print(f"Average Convergence (Final): {results['average_convergence'][-1]:.2f}")
    print(f"Best Length: {results['best_length']:.2f}")
    if results['optimal_length'] is not None:
        print(f"Optimal Length: {results['optimal_length']:.2f}")
    mre = calculate_mre([results['average_length']], results['optimal_length']) if results['optimal_length'] is not None else None
    if mre is not None:
        print(f"Mean Relative Error (MRE): {mre:.2f}%")
    print(f"Average Discovery Epoch: {results['discovery_epoch']:.2f}")
    print(f"Standard Deviation of Lengths: {results['standard_deviation_length']:.2f}")
    
    # print results in single line: Instance & Avg. Len. & Avg. Time & Best & Optimal & MRE & Std. Dev for table in the report
    #print(f"{tsp_file.split('/')[1]} & {results['average_length']:.2f} & {results['average_time']:.2f} & {results['best_length']:.2f} & {results['optimal_length']:.2f} & {mre:.2f} & {results['standard_deviation_length']:.2f} \\")

    # print results in single line: Instance & Best & Avg. Len. & Std. Dev. & Optimal & MRE(%) & Time (s) & Conv. Iter for table in the report
    #print(f"{tsp_file} & {results['best_length']:.2f} & {results['average_length']:.2f} & {results['standard_deviation_length']:.2f} & {results['optimal_length']:.2f} & {mre:.2f}\% & {results['average_time']:.2f} & {results['discovery_epoch']:.2f} \\")

    # print single line in this format: Avg. Len. & Std. Dev. & MRE for table in the report
    #print(f"{results['average_length']:.2f} & {results['standard_deviation_length']:.2f} & {mre:.2f}\% \\")
