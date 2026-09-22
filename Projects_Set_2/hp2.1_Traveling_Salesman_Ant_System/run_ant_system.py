# Daniel Peña Fonseca

###########################
#
# Ant System (AS) for TSP
#
###########################

import math
import numpy as np
import matplotlib.pyplot as plt
import random



###############################################################
## To do: Write the initialize_pheromone_levels function:
###############################################################

def initialize_pheromone_levels(number_of_cities, tau_0):
   
    number_of_cities = int(number_of_cities)
    pheromone_matrix = tau_0 * np.ones((number_of_cities, number_of_cities))
    
    return pheromone_matrix



###############################################################
## To do: Write the get_visibility function:
###############################################################

def get_visibility(city_locations):

    city_locations = np.array(city_locations)
    number_of_cities = len(city_locations)
    distance_matrix = np.zeros((number_of_cities, number_of_cities))

    for index_1 in range(number_of_cities):
       for index_2 in range(number_of_cities):
          distance_matrix[index_1][index_2] = np.linalg.norm(np.subtract(city_locations[index_1], city_locations[index_2]))
          
    # To avoid distances of 0 length, as occurs for the diagonals, we set them as 1. The number does not matter,
    # since we will not be working with the possibility of going from one city to the same city in the same step.
    nonzero_unity_matrix = np.identity(number_of_cities)
    distance_matrix += nonzero_unity_matrix
    visibility_matrix = 1/distance_matrix
    
    return visibility_matrix



def get_node(taboo_list, pheromone_levels, visibility, alpha, beta):

    number_of_cities = int(np.sqrt(pheromone_levels.size))
    all_cities_list = list(range(number_of_cities))
    possible_list = np.delete(all_cities_list, taboo_list)
    
    current_city = taboo_list[-1]

    probabilities = np.zeros(len(possible_list))

    normalizing_sum = 0
    for index_city in possible_list:
        normalizing_sum += (pheromone_levels[current_city][index_city]** alpha) * (visibility[current_city][index_city]** beta)

    i = 0
    for index_city in possible_list:
        probabilities[i] = (1/normalizing_sum) * (pheromone_levels[current_city][index_city]** alpha) * (visibility[current_city][index_city]** beta)
        i+=1

    indices_and_probability_list = np.stack((possible_list, probabilities), axis = 1)
    sorted_indices_and_probability_list = np.array(sorted(indices_and_probability_list, key = lambda x:x[1]))

    sorted_probability_list=[]
    for i in range(len(indices_and_probability_list)):
       sorted_probability_list.append(sorted_indices_and_probability_list[i][1])

    sorted_probability_list = np.cumsum(np.array(sorted_probability_list))

    r = random.random()
    # Using argmax I get the index of the list where the first element greater than r is, like we did in Roulette-Wheel_Selection
    chosen_index_in_sorted_list = np.argmax(sorted_probability_list>r)
    chosen_city = sorted_indices_and_probability_list[chosen_index_in_sorted_list][0]

    return int(chosen_city)



#################################################################
## To do: Write the generate_path function (Note: You may wish
##       to add more functions, e.g., get_node. That is allowed).
#################################################################

def generate_path(pheromone_levels, visibility, alpha, beta):

    number_of_cities = int(np.sqrt(pheromone_levels.size))
    starting_city = random.randint(0,number_of_cities - 1)

    taboo_list = [starting_city]

    for i in range(1,number_of_cities):
        new_node = get_node(taboo_list, pheromone_levels, visibility, alpha, beta)
        taboo_list.append(new_node)

    return taboo_list



###############################################################
## To do: Write the get_path_length function:
###############################################################

def get_path_length(path, city_locations):

    list_of_lengths = np.zeros(len(path)+1)
    for i in path:
        list_of_lengths[i-1] = np.linalg.norm(np.subtract(city_locations[i], city_locations[i-1]))

    list_of_lengths += np.linalg.norm(np.subtract(city_locations[path[-1]], city_locations[path[0]]))

    path_length = np.sum(np.array(list_of_lengths))

    return path_length


def get_path_length(path, city_locations):

    path_length = 0

    for i in range(1, len(path)):
        from_city = path[i-1]
        to_city = path[i]
        path_length += np.linalg.norm(np.subtract(city_locations[from_city], city_locations[to_city]))
  
    path_length += np.linalg.norm(np.subtract(city_locations[path[-1]], city_locations[path[0]]))

    return path_length
    


###############################################################
## To do: Write the compute_delta_pheromone_levels function:
###############################################################

def compute_delta_pheromone_levels(path_collection, path_length_collection):
    
    number_of_ants = len(path_collection)
    number_of_cities = len(path_collection[0])

    delta_pheromone_levels = np.zeros((number_of_cities, number_of_cities))

    for index_ant in range(number_of_ants):
        # Adding the first city to the path. In path_length this last journey is already included
        first_city = path_collection[index_ant][0]
        path = path_collection[index_ant][:]
        path_length = path_length_collection[index_ant]
        path.append(first_city)

        for index_city in range(len(path)-1):
            from_city = path[index_city]
            to_city = path[index_city+1]

            delta_pheromone_levels[from_city][to_city] += 1/path_length

    return delta_pheromone_levels



###############################################################
## To do: Write the update_pheromone_levels function:
###############################################################

def update_pheromone_levels(pheromone_levels, delta_pheromone_levels, rho):

    new_pheromone_levels = (1-rho) * np.array(pheromone_levels) + np.array(delta_pheromone_levels)
    new_pheromone_levels_in_interval = np.clip(new_pheromone_levels, a_min = 10**(-15), a_max = None)

    return new_pheromone_levels_in_interval



##################################################
#  Plots the cities (nodes):
##################################################

def plot_cities(plt, city_locations):
  x = []
  y = []
  for city_index in range(len(city_locations)):
    x.append(city_locations[city_index][0])
    y.append(city_locations[city_index][1])
  plt.scatter(x,y,zorder=1,color='yellow')



##################################################
#  Plots the path (connections between nodes):
##################################################

def plot_path(plt, path):

  connections_x = []
  connections_y = []
  for index in path:
    location_x = city_locations[index][0]
    connections_x.append(location_x)
    location_y = city_locations[index][1]
    connections_y.append(location_y)
  start_location_x = city_locations[path[0]][0]
  start_location_y = city_locations[path[0]][1]
  connections_x.append(start_location_x)
  connections_y.append(start_location_y)
  plt.plot(connections_x,connections_y,color='lime',zorder=0)



#####################################
# Main program:
#####################################

###########################
# Data:
###########################
from city_data import city_locations
number_of_cities = len(city_locations)



###########################
# Parameters:
###########################
number_of_ants = 50 ## Changes allowed.
alpha = 1.0         ## Changes allowed.
beta = 5.0          ## Changes allowed.
rho = 0.5           ## Changes allowed.
tau_0 = 0.1         ## Changes allowed.

target_path_length = 99.9999999



#################################
# Initialization:
#################################

seed = 1687
random.seed(seed)
print(f'Seed used: {seed} \n')

plot_range = 20
number_of_iterations = 50
pause_interval = 0.05

# Load data
from city_data import city_locations
number_of_cities = len(city_locations)

# Generate a random path:
path = np.random.permutation(number_of_cities) # https://numpy.org/doc/stable/reference/random/generated/numpy.random.permutation.html

# Prepare plot
plt.xlim(0,plot_range)
plt.ylim(0,plot_range)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
ax.set_facecolor('xkcd:black')


pheromone_levels = initialize_pheromone_levels(number_of_cities, tau_0)
visibility = get_visibility(city_locations)



#################################
# Main loop:
#################################

iteration_index = 0
minimum_path_length = math.inf
path_length = math.inf


while (minimum_path_length > target_path_length):

  iteration_index += 1
  path_collection = []
  path_length_collection = []

  for ant_index in range(number_of_ants):  

    # Generate paths:
    path = generate_path(pheromone_levels, visibility, alpha, beta) 
    path_length = get_path_length(path, city_locations) 

    if (path_length < minimum_path_length):
      minimum_path_length = path_length
      print(f'Best path found yet: {path}')
      print(f'Path length: {minimum_path_length}\n' )
      
      # To do: Add code for plotting here
      plt.cla()
      plot_cities(plt,city_locations)
      plot_path(plt,path)
      plt.pause(pause_interval)
      plt.show(block=False)
      plt.ion()

    path_collection.append(path)
    path_length_collection.append(path_length)

  # Update pheromone levels:
  delta_pheromone_levels = compute_delta_pheromone_levels(path_collection,path_length_collection) 
  pheromone_levels = update_pheromone_levels(pheromone_levels, delta_pheromone_levels, rho) 

print(f'Best path for current run: {path}')
print(f'Path length: {minimum_path_length}')
input(f'Press return to exit')