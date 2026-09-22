

import numpy as np
import random
import copy
import threading
import sys

from function_data import load_function_data


seed = 3061

print('Random seed:', seed)
random.seed(seed)
np.random.seed(seed)


def one_instruction_decoding(index_of_operator, operand_1, operand_2):
    if index_of_operator == 1:
        new_destination_register = operand_1 + operand_2
    
    if index_of_operator == 2:
        new_destination_register = operand_1 - operand_2

    if index_of_operator == 3:
        new_destination_register = operand_1 * operand_2

    if index_of_operator == 4:
        if operand_2 == 0:
            new_destination_register = 5000
        else:
            new_destination_register = operand_1 / operand_2

    return new_destination_register


def initialization(number_of_chromosomes, number_of_instructions, number_of_constant_registers, number_of_variable_registers, number_of_operators):
    
    # population will be a list of arrays, arrays called chromosomes
    # Each individual will be a list of arrays, arrays called instructions
    # Each instruction will be a list of elements, elements called genes

    number_of_all_registers = number_of_constant_registers + number_of_variable_registers

    population = []

    for chromosome_index in range(number_of_chromosomes):
        chromosome = []
        for instruction_index in range(number_of_instructions):
                
            operator = random.randint(1, number_of_operators)
            destination_register = random.randint(1, number_of_variable_registers)
            operand_1 = random.randint(1, number_of_all_registers)
            operand_2 = random.randint(1, number_of_all_registers)

            chromosome.append([operator, destination_register, operand_1, operand_2])
        
        population.append(chromosome)
                
    return population


def decoding_one_chromosome(x_value, chromosome, number_of_variable_registers, constant_registers):
 
    # Initialization of variable registers setting the first one as x, the rest as 0
    variable_registers = [0] * number_of_variable_registers
    variable_registers[0] = x_value
    for i in range(1, number_of_variable_registers):
        variable_registers[i] = 0

    all_registers = variable_registers + constant_registers

    number_of_instructions = len(chromosome)
    for instruction_index in range(number_of_instructions):

        index_of_operator = chromosome[instruction_index][0]
        index_of_destination_register = chromosome[instruction_index][1] -1 
        index_of_operand_1 = chromosome[instruction_index][2] - 1
        index_of_operand_2 = chromosome[instruction_index][3] - 1

        operand_1 = all_registers[index_of_operand_1]
        operand_2 = all_registers[index_of_operand_2]

        destination_register = one_instruction_decoding(index_of_operator, operand_1, operand_2)
        all_registers[index_of_destination_register] = destination_register


    chromosome_function_in_x_value = all_registers[0]
    
    return chromosome_function_in_x_value


def chromosome_evaluation(x_data, function_data, chromosome, constant_registers, number_of_variable_registers):
    # To measure which chromosome has a better fitness function, we check how accurate the function defined by each chromosome 
    # approximates the data.

    x_data = np.asarray(x_data)
    function_data = np.asarray(function_data)

    chromosome_results = []

    for x_value in x_data:
        chromosome_function_in_x_value = decoding_one_chromosome(x_value, chromosome, number_of_variable_registers, constant_registers)
        chromosome_results.append(chromosome_function_in_x_value)

    chromosome_results = np.array(chromosome_results, dtype = float)
    
    error_sum = np.sqrt(np.mean((function_data - chromosome_results)**2))
    
    if error_sum != 0:
        fitness_value = 1/error_sum
        # Length penalty only for those with length greater than 75
        penalty = max(1, len(chromosome)/75)
        fitness_value = fitness_value/penalty

    else:
        fitness_value = np.inf

    return fitness_value


def tournament_selection(fitness_list, tournament_probability, tournament_size):

    enumerated_fitness_list = list(enumerate(fitness_list))
    
    tournament_fitness_list = random.sample(enumerated_fitness_list, tournament_size)
    sorted_tournament_fitness_list = sorted(tournament_fitness_list, key = lambda x:x[1], reverse = True)

    winner_index = 0
    selected = False
    while winner_index < tournament_size and selected == False:

        r = random.random()
        if r < tournament_probability:
            selected = True

        else:
            winner_index += 1

    if winner_index == tournament_size:
        winner_index = tournament_size - 1

    return sorted_tournament_fitness_list[winner_index][0]


def crossover(chromosome_1, chromosome_2):

    # For the LGP, we have to do crossover in two points. Also, we cannot do crossover that cuts
    # an instruction; it has to be between them.

    number_of_instructions_chromosome_1 = len(chromosome_1)
    number_of_instructions_chromosome_2 = len(chromosome_2)

    cross_point_1_chromosome_1 = random.randint(1, number_of_instructions_chromosome_1-1)
    cross_point_2_chromosome_1 = random.randint(1, number_of_instructions_chromosome_1-1)
    lowest_crossover_point_chromosome_1 = min(cross_point_1_chromosome_1, cross_point_2_chromosome_1)
    highest_crossover_point_chromosome_1 = max(cross_point_1_chromosome_1, cross_point_2_chromosome_1)

    cross_point_1_chromosome_2 = random.randint(1, number_of_instructions_chromosome_2-1)
    cross_point_2_chromosome_2 = random.randint(1, number_of_instructions_chromosome_2-1)
    lowest_crossover_point_chromosome_2 = min(cross_point_1_chromosome_2, cross_point_2_chromosome_2)
    highest_crossover_point_chromosome_2 = max(cross_point_1_chromosome_2, cross_point_2_chromosome_2)

    if (lowest_crossover_point_chromosome_1 == highest_crossover_point_chromosome_1) or (lowest_crossover_point_chromosome_2 == highest_crossover_point_chromosome_2):
        new_chromosome_1, new_chromosome_2 = chromosome_1, chromosome_2

    else:
        new_chromosome_1 = chromosome_1[:lowest_crossover_point_chromosome_1] + chromosome_2[lowest_crossover_point_chromosome_2: highest_crossover_point_chromosome_2] + chromosome_1[highest_crossover_point_chromosome_1:]
        new_chromosome_2 = chromosome_2[:lowest_crossover_point_chromosome_2] + chromosome_1[lowest_crossover_point_chromosome_1: highest_crossover_point_chromosome_1] + chromosome_2[highest_crossover_point_chromosome_2:]

    return new_chromosome_1, new_chromosome_2


def randomize_one_element(element_index, number_of_operators, number_of_variable_registers, number_of_constant_registers):

    number_of_all_registers = number_of_variable_registers + number_of_constant_registers

    if element_index == 0:
        # Operator index
        new_index = random.randint(1, number_of_operators)
    
    if element_index == 1:
        # Destination register
        new_index = random.randint(1, number_of_variable_registers)

    if element_index == 2:
        # Operand 1
        new_index = random.randint(1, number_of_all_registers)

    if element_index == 3:
        # Operand 2
        new_index = random.randint(1, number_of_all_registers)

    return new_index


def mutate(chromosome, mutation_probability, number_of_operators, number_of_variable_registers, number_of_constant_registers):

    number_of_instructions = len(chromosome)

    for instruction_index in range(number_of_instructions):
        for element_index in range(4):
            r = random.random()
            if r < mutation_probability:
                new_index = randomize_one_element(element_index, number_of_operators, number_of_variable_registers, number_of_constant_registers)
                chromosome[instruction_index][element_index] = new_index

    return chromosome


def mutation_probability_change(previous_mutation_probability, all_the_maximum_fitnesses, number_of_instructions, time_window_size, probability_change_factor):

    initial_mutation_probability = 1/(number_of_instructions*4)

    if len(all_the_maximum_fitnesses) < time_window_size:
        new_mutation_probability = previous_mutation_probability

    else:
        fitness_difference = all_the_maximum_fitnesses[-1] - all_the_maximum_fitnesses[-time_window_size]

        # When there is a significant improvement, we create randomness for at least time_window_size times
        # so that the system has more variety to work with:
        if fitness_difference > 1e-6:
            new_mutation_probability = previous_mutation_probability * probability_change_factor

        elif fitness_difference < 1e-12:
            new_mutation_probability = initial_mutation_probability * 1.5

        else:
            # If there has been improvement, we return to the previous 
            new_mutation_probability = initial_mutation_probability

    return new_mutation_probability




# I am going to use flags so that I can save the best chromosome whenever I want.
# When pressing s + Enter, the program saves the chromosome and keeps running
# When pressing Enter, the program saves the chromosome and stops running
def save_best_chromosome(best_chromosome):
    with open("best_chromosome.py", "w") as f:
        f.write(f"best_chromosome = {repr(best_chromosome)}\n")
    print("\n Best chromosome saved to best_chromosome.py")

def wait_for_input(stop_flag, save_flag):
    while True:
        key = input()  
        if key.lower() == "s":
            save_flag["save"] = True  
        else:
            stop_flag["stop"] = True  
            break



def lgp(x_data, function_data, constant_registers, number_of_chromosomes, maximum_chromosome_length, number_of_instructions, number_of_variable_registers, number_of_operators,
        number_of_generations, tournament_probability, tournament_size, crossover_probability, mutation_probability, time_window_size, probability_change_factor):
    
    old_mutation_probability = mutation_probability

    number_of_constant_registers = len(constant_registers)
    # Initialize population
    population = initialization(number_of_chromosomes, number_of_instructions, number_of_constant_registers, number_of_variable_registers, number_of_operators)

    all_the_maximum_fitnesses =[]

    stop_flag = {"stop": False}
    save_flag = {"save": False}
    listener = threading.Thread(target=wait_for_input, args=(stop_flag, save_flag), daemon=True)
    listener.start()
    
    for generation_index in range(number_of_generations):

        # Since the fitness will always be greater > 0, I can initialize it as 0
        maximum_fitness = 0 
        best_chromosome = None
        
        fitness_list = []
        
        for chromosome in population:
            fitness = chromosome_evaluation(x_data, function_data, chromosome, constant_registers, number_of_variable_registers)
            
            if (fitness > maximum_fitness):
                maximum_fitness = fitness
                best_chromosome = copy.deepcopy(chromosome)
            fitness_list.append(fitness)

        temporary_population = []

        all_the_maximum_fitnesses.append(maximum_fitness)

        for i in range(0,number_of_chromosomes,2):

            index_1 = tournament_selection(fitness_list, tournament_probability, tournament_size)
            index_2 = tournament_selection(fitness_list, tournament_probability, tournament_size)
            chromosome_1 = copy.deepcopy(population[index_1])
            chromosome_2 = copy.deepcopy(population[index_2])

            r = random.random()
            if r < crossover_probability:
                new_chromosome_1, new_chromosome_2 = crossover(chromosome_1,chromosome_2)
                
            else:
                new_chromosome_1, new_chromosome_2 = chromosome_1, chromosome_2
                

            while len(new_chromosome_1) > maximum_chromosome_length or len(new_chromosome_2) > maximum_chromosome_length:
                index_1 = tournament_selection(fitness_list, tournament_probability, tournament_size)
                index_2 = tournament_selection(fitness_list, tournament_probability, tournament_size)
                chromosome_1 = copy.deepcopy(population[index_1])
                chromosome_2 = copy.deepcopy(population[index_2])

                r = random.random()
                if r < crossover_probability:
                    new_chromosome_1, new_chromosome_2 = crossover(chromosome_1,chromosome_2)
                    
                else:
                    new_chromosome_1, new_chromosome_2 = chromosome_1, chromosome_2

            temporary_population.append(new_chromosome_1)
            temporary_population.append(new_chromosome_2) 

        for i in range(number_of_chromosomes):
            original_chromosome = temporary_population[i]

            new_mutation_probability = mutation_probability_change(old_mutation_probability, all_the_maximum_fitnesses, number_of_instructions, time_window_size, probability_change_factor)

            mutated_chromosome = mutate(original_chromosome, new_mutation_probability, number_of_operators, number_of_variable_registers, number_of_constant_registers)
            temporary_population[i] = mutated_chromosome

        old_mutation_probability = new_mutation_probability
        temporary_population[0] = copy.deepcopy(best_chromosome)
        population = copy.deepcopy(temporary_population)

        if generation_index % 100 == 0:
            print(f'Generation {generation_index}: {maximum_fitness}')

        if save_flag["save"]:
            save_best_chromosome(best_chromosome)
            print('Code will keep running')
            save_flag["save"] = False

        if stop_flag["stop"]:
            save_best_chromosome(best_chromosome)
            return maximum_fitness, best_chromosome
            

    # We always save the best at the end
    save_best_chromosome(best_chromosome)

    return maximum_fitness, best_chromosome 






constant_registers = [1.0, -1.0, 2.0, 3.0, 4.0]
number_of_chromosomes = 100
maximum_chromosome_length = 100
number_of_instructions = 25
number_of_variable_registers = 7
number_of_operators = 4

number_of_generations = 100000
tournament_probability = 0.7

crossover_probability = 0.8
mutation_probability = 1/(number_of_instructions*4)

time_window_size = 5
probability_change_factor = 1.2

x_data = []
function_data = []

samples = load_function_data()
for i in range(len(samples)):
    x_data.append(samples[i][0])
    function_data.append(samples[i][1])

tournament_size = 10

best_chromosome = None
try:
    maximum_fitness, best_chromosome = lgp(x_data, function_data, constant_registers, number_of_chromosomes, maximum_chromosome_length, number_of_instructions, number_of_variable_registers, number_of_operators,
            number_of_generations, tournament_probability, tournament_size, crossover_probability, mutation_probability, time_window_size, probability_change_factor)
except Exception as e:
    print(f"An error occurred: {e}")
    if 'best_chromosome' != None:
        save_best_chromosome(best_chromosome)
    sys.exit(1)


    