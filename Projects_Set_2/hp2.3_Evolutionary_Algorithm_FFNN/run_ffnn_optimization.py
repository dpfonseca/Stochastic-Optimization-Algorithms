# Daniel Peña Fonseca

import numpy as np
import random
import copy
from slopes import get_slope_angle
from run_encoding_decoding_test import decode_chromosome
from truck_model import truck_model_class




truck = truck_model_class(
    maximum_temperature = 750,
    mass = 20000,  
    constant_tau = 30,
    constant_c_h = 40,
    ambient_temperature = 283,
    constant_c_b = 3000
)


def sigmoid(x, c = 1):
    result = 1/(1+np.exp(-c*x))
    return result


def ffnn_pass(w_ih, w_ho, input_list):

    # We include the bias
    input_list = np.array([1] + input_list)
    w_ih = np.array(w_ih)
    w_ho = np.array(w_ho)

    result_hidden = sigmoid(w_ih @ input_list)

    new_input_list = np.array([1] + result_hidden.tolist())
    result_output = sigmoid(w_ho @ new_input_list)

    return result_output


def initialization(number_of_chromosomes, number_of_genes):

    population = []
    for chromosome_index in range(number_of_chromosomes):
        new_chromosome = []
        for genes_index in range(number_of_genes):
            new_gene = random.random()
            new_chromosome.append(new_gene)

        population.append(new_chromosome)

    return population


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


def crossover(chromosome1, chromosome2):

    number_of_genes = len(chromosome1)

    cross_point = random.randint(1, number_of_genes-1)

    new_chromosome_1 = chromosome1[:cross_point] + chromosome2[cross_point:]
    new_chromosome_2 = chromosome2[:cross_point] + chromosome1[cross_point:]
    return [new_chromosome_1, new_chromosome_2]


def mutate(chromosome, mutation_probability):

    number_of_genes = len(chromosome)

    for gene_index in range(number_of_genes):
        r = random.random()
        if r < mutation_probability:
            chromosome[gene_index] = random.random()

    return chromosome


def chromosome_evaluation_in_slope(chromosome, data_set_index, slope_index, road_length, minimum_velocity, maximum_velocity, maximum_angle, maximum_temperature, time_step, number_of_hidden_neurons, w_max):

    # Evaluating the chromosome requires using a FFNN; therefore, we need to decode the chromosome to 
    # obtain the weights
    n_i = 3
    n_h = number_of_hidden_neurons
    n_o = 2
    [w_ih, w_ho] = decode_chromosome(chromosome, n_i, n_h, n_o, w_max)

    # Variable initialization:
    truck.position = 0.0
    truck.velocity = 20
    truck.angle = get_slope_angle(truck.position, slope_index, data_set_index)
    truck.gear = 7
    truck.temperature = 500

    # Amount of time steps in 2 seconds: 
    time_steps_in_2_seconds = int(2/time_step)
    change_gear_evolution = []
    
    # FFNN
    velocity_fraction = truck.velocity/maximum_velocity
    angle_fraction = truck.angle/maximum_angle
    temperature_fraction = truck.temperature/maximum_temperature
    
    input_list = [velocity_fraction, angle_fraction, temperature_fraction]
    ffnn_output = ffnn_pass(w_ih, w_ho, input_list)

    gear_output = ffnn_output[0]
    # Since the sigmoid function returns outputs from 0 to 1:
    if gear_output <= 1/3:
        change_gear = -1

    elif gear_output > 2/3:
        change_gear = 1

    else:
        change_gear = 0

    truck.gear += change_gear
    change_gear_evolution.append(abs(change_gear))

    truck.pedal_pressure = ffnn_output[1]

    truck.velocity = truck.velocity_change(time_step, truck.velocity)
    if truck.velocity > maximum_velocity:
        fitness = 0
        print('Stopped at x = 0')
        return fitness

    time = 0
    while truck.position < road_length:

        # Using the previous velocity, we calculate position, angle, temperature
        previous_position = truck.position
        truck.position += truck.velocity * time_step
        truck.angle = get_slope_angle(truck.position, slope_index, data_set_index)

        if truck.position >= road_length:

            truck.position = road_length
            last_step_distance = road_length - previous_position
            last_step_time = last_step_distance/truck.velocity
            time += last_step_time

            average_velocity = truck.position/time
            fitness = truck.position * average_velocity
            break
    
        truck.temperature = truck.temperature + truck.temperature_change(time_step)
        if truck.temperature < truck.ambient_temperature:
            truck.temperature = truck.ambient_temperature

        # Using the velocity when inputting the current state, the angle and the temperature, we calculate the new gear change and pedal pressure:

        # FFNN
        velocity_fraction = truck.velocity/maximum_velocity
        angle_fraction = truck.angle/maximum_angle
        temperature_fraction = truck.temperature/maximum_temperature

        input_list = [velocity_fraction, angle_fraction, temperature_fraction]
        ffnn_output = ffnn_pass(w_ih, w_ho, input_list)

        gear_output = ffnn_output[0]
        # Since the sigmoid function returns outputs from 0 to 1:
        if gear_output <= 1/3:
            change_gear = -1

        elif gear_output > 2/3:
            change_gear = 1

        else:
            change_gear = 0

        # Updating
        # I check if there has been any gear changes in the past 2 seconds
        if len(change_gear_evolution) < time_steps_in_2_seconds:
            gear_max = max(change_gear_evolution)
    
        else:
            gear_max = max(change_gear_evolution[-time_steps_in_2_seconds:])

        if gear_max == 0:
            if (truck.gear == 1) and (change_gear == -1):
                change_gear = 0
                truck.gear = 1

            elif (truck.gear == 9) and (change_gear == 1):
                change_gear = 0
                truck.gear = 9
                
        else:
            change_gear = 0
        
        truck.gear += change_gear
        change_gear_evolution.append(abs(change_gear))
        
        truck.pedal_pressure = ffnn_output[1]

        # With all the variables updated, we calculate the velocity that will get the truck to its next point
        truck.velocity = truck.velocity_change(time_step, truck.velocity)

        # If any constraint is violated, we terminate the process
        if (truck.velocity < minimum_velocity) or (truck.velocity > maximum_velocity) or (truck.temperature > maximum_temperature):
            if time != 0:
                average_velocity = truck.position/time
                fitness = truck.position * average_velocity
            if time == 0:
                fitness = 0
            #print(f'Stopped at x = {truck.position}')
            return fitness

        time += time_step

    #print(f'Stopped at x = {truck.position}')

    return fitness


def chromosome_evaluation_in_set(chromosome, data_set_index, road_length, minimum_velocity, maximum_velocity, maximum_angle, maximum_temperature, time_step, number_of_hidden_neurons, w_max):

    # With this function we repeat the process of evaluating slope, for all the slopes in a set.

    # Training set
    if data_set_index == 1:
        number_of_slopes = 10

    # Validation set or Test set
    if data_set_index == 2 or data_set_index == 3:
        number_of_slopes = 5

    fitness_list = []
    for slope_index in range(1, number_of_slopes+1):
        new_fitness = chromosome_evaluation_in_slope(chromosome, data_set_index, slope_index, road_length, minimum_velocity, maximum_velocity, maximum_angle, maximum_temperature, time_step, number_of_hidden_neurons, w_max)
        fitness_list.append(new_fitness)
    
    chromosome_fitness_in_set = sum(fitness_list)/len(fitness_list)

    return chromosome_fitness_in_set


def ffnn(number_of_chromosomes, number_of_genes, number_of_generations, tournament_probability, tournament_size, crossover_probability, mutation_probability, road_length, minimum_velocity, maximum_velocity, maximum_angle, maximum_temperature, time_step, number_of_hidden_neurons, w_max):

    # Initialization
    population = initialization(number_of_chromosomes, number_of_genes)

    training_best_fitness = 0 
    training_best_chromosome = None
    validation_best_chromosome = None
    validation_best_fitness = 0
    validation_best_fitness_list = []
    training_best_fitness_list = []

    for generation_index in range(number_of_generations):
        
        
        fitness_list = []

        # Evaluation
        for chromosome in population:
            # Evaluation in training set
            data_set_index = 1
            training_fitness = chromosome_evaluation_in_set(chromosome, data_set_index, road_length, minimum_velocity, maximum_velocity, maximum_angle, maximum_temperature, time_step, number_of_hidden_neurons, w_max)

            if (training_fitness > training_best_fitness):
                training_best_fitness = training_fitness
                training_best_chromosome = chromosome.copy()
            fitness_list.append(training_fitness)

            # Evaluation in validation set
            data_set_index = 2
            validation_fitness = chromosome_evaluation_in_set(chromosome, data_set_index, road_length, minimum_velocity, maximum_velocity, maximum_angle, maximum_temperature, time_step, number_of_hidden_neurons, w_max)
            if (validation_fitness > validation_best_fitness):
                validation_best_fitness = validation_fitness
                validation_best_chromosome = chromosome.copy()
            

        temporary_population = []
        

        for i in range(0,number_of_chromosomes,2):

            index_1 = tournament_selection(fitness_list, tournament_probability, tournament_size)
            index_2 = tournament_selection(fitness_list, tournament_probability, tournament_size)
            chromosome_1 = population[index_1].copy()
            chromosome_2 = population[index_2].copy()

            r = random.random()
            if r < crossover_probability:
                new_chromosome_1, new_chromosome_2 = crossover(chromosome_1,chromosome_2)
                
            else:
                new_chromosome_1, new_chromosome_2 = chromosome_1, chromosome_2

            temporary_population.append(new_chromosome_1)
            temporary_population.append(new_chromosome_2) 

        for i in range(number_of_chromosomes):
            new_chromosome = mutate(temporary_population[i], mutation_probability)
            temporary_population[i] = new_chromosome

        temporary_population[0] = training_best_chromosome.copy()
        population = copy.deepcopy(temporary_population)

        print(f'Generation {generation_index+1}: Training best fitness = {training_best_fitness}, Validation best fitness = {validation_best_fitness}')

        training_best_fitness_list.append(training_best_fitness)
        validation_best_fitness_list.append(validation_best_fitness)
        

    return validation_best_chromosome, validation_best_fitness, validation_best_fitness_list, training_best_fitness_list

if __name__ == "__main__":

    seed = random.randint(0,10000)
    seed = 6723
    print('Main -> Random seed:', seed)
    random.seed(seed)
    np.random.seed(seed)

    # Parameters
    number_of_chromosomes = 30

    number_of_inputs = 3
    number_of_hidden_neurons = 4
    number_of_outputs = 2
    number_of_genes = (number_of_inputs + 1) * number_of_hidden_neurons + (number_of_hidden_neurons + 1) * number_of_outputs

    number_of_generations = 100
    tournament_probability = 0.8
    tournament_size = 4
    crossover_probability = 0.7
    time_step_size = 0.1
    road_length = 1000.0
    mutation_probability = 1/number_of_genes
    w_max = 10

    maximum_velocity = 25
    minimum_velocity = 1
    maximum_temperature = 750
    maximum_angle = 10


    validation_best_chromosome, validation_best_fitness, validation_best_fitness_list, training_best_fitness_list = ffnn(number_of_chromosomes, number_of_genes, number_of_generations, tournament_probability, tournament_size, crossover_probability, mutation_probability, road_length, minimum_velocity, maximum_velocity, maximum_angle, maximum_temperature, time_step_size, number_of_hidden_neurons, w_max)

    with open("best_chromosome.py", "w") as f:
        f.write(f"best_chromosome = {validation_best_chromosome}\n")

    # Save fitness lists
    with open("data_fitness_evolution.py", "w") as f:
        f.write(f"validation_best_fitness_list = {validation_best_fitness_list}\n")
        f.write(f"training_best_fitness_list = {training_best_fitness_list}\n")


