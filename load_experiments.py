import os

def return_experiments():
    experiments = []
    for file in os.listdir("res"):
        if file.endswith(".tsp"):
            tsp_file = os.path.join("res", file)
            tour_file = os.path.join("res", file.replace(".tsp", ".opt.tour"))
            experiments.append((tsp_file, tour_file))
    return experiments

def return_experiments_with_tours():
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

