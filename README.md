# 4p96-project
Ant Colony Optimization for the Traveling Salesman Problem

For this project, we will be utilizing ant-like swarm intelligence in order to try and find optimal solutions to
the well-known Traveling Salesman Problem. In ACO, artificial ants construct solutions by probabilistically
selecting the next city to visit based on pheromone intensity and heuristic desirability. The iterative process of
optimization encourages ants to converge toward shorter paths over time, allowing the colony to act as a method
of optimal pathfinding.
There are a number of benchmark TSP datasets available; several have been linked in one place by Ohlmann
and Thomas, and we will also be using the more ”standard” TSPLib, which contains a huge number of
varyingly sized examples. These all contain predefined sets of cities and their optimal distances, allowing
performance comparisons against known solutions.
Several experiments will be conducted to evaluate the performance of the implemented ACO algorithm.
First, the project will analyze the impact of the key ACO parameters: number of ants, pheromone evaporation
rate, influence of pheromones, and the influence of heuristic visibility. Different parameter combinations will
be tested to observe their effect on convergence speed and final path length. Then, the algorithms will be tested
across multiple TSP instances to evaluate best path lengths obtained, average path lengths across runs, and
convergence behavior over time; performances will be compared against best known solutions where they are
available.
