import math
from genetic_algorithm import run_function_optimization

number_of_runs = 100                
population_size = 100               
maximum_variable_value = 5          # (x_i in [-a,a], where a = maximumVariableValue)
number_of_genes = 50                
number_of_variables = 2  	    
tournament_size = 2                 
tournament_probability = 0.75       
crossover_probability = 0.8         
number_of_generations = 2000        


mutation_probability_list = [0, 0.005, 0.01, 0.02, 0.05, 0.1]


def median(x_values):
    
    x_values_sorted = sorted(x_values)
    even_or_odd = len(x_values) % 2

    if even_or_odd == 1:
        index = int(math.floor(len(x_values)/2))
        median = x_values_sorted[index]

    elif even_or_odd == 0:
        index_1 = int(len(x_values)/2)
        index_2 = index_1-1
        median = (x_values_sorted[index_1]+x_values_sorted[index_2])/2

    return median


maximum_list =[]
for mutation_probability in mutation_probability_list:
    for run_index in range(number_of_runs):
        [maximum_fitness, x_best] = run_function_optimization(population_size, number_of_genes, number_of_variables, maximum_variable_value, tournament_size, \
                                       tournament_probability, crossover_probability, mutation_probability, number_of_generations)
        maximum_list.append(maximum_fitness)
    maximum_list_median = median(maximum_list)

    print(f'Mutation probability: {mutation_probability}, Median: {maximum_list_median:.4f}')