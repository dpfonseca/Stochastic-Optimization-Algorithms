import math
from genetic_algorithm import run_function_optimization

population_size = 100               
maximum_variable_value = 5          # Do NOT change: (x_i in [-a,a], where a = maximumVariableValue)
number_of_genes = 50                
number_of_variables = 2  	    

tournament_size = 2                 
tournament_probability = 0.75       
crossover_probability = 0.8         
mutation_probability = 0.02         
number_of_generations = 2000        

[maximum_fitness, x_best] = run_function_optimization(population_size, number_of_genes, number_of_variables, maximum_variable_value, tournament_size, \
                                       tournament_probability, crossover_probability, mutation_probability, number_of_generations)
output = f"Fitness: {maximum_fitness:.4f}, x = ({x_best[0]:.8f},{x_best[1]:.8f})"
print(output)



