import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from best_chromosome import best_chromosome as chromosome
from run_encoding_decoding_test import decode_chromosome
from slopes import get_slope_angle
from truck_model import truck_model_class
from run_ffnn_optimization import ffnn_pass

truck = truck_model_class(
    maximum_temperature = 750,
    mass = 20000,  
    constant_tau = 30,
    constant_c_h = 40,
    ambient_temperature = 283,
    constant_c_b = 3000
)

# For the test, I will modify the function chromosome_evaluation_in_slope from run_ffnn_optimization
# so that now, instead of returning the fitness of the individual, it returns the
# evolution of the variables indicated.

def evolution_in_slope(chromosome, data_set_index, slope_index, road_length, minimum_velocity, maximum_velocity, maximum_angle, maximum_temperature, time_step, number_of_hidden_neurons, w_max):

    # Evaluating the chromosome requires using a FFNN; therefore, we need to decode the chromosome to 
    # obtain the weights
    n_i = 3 # inputs
    n_h = number_of_hidden_neurons # hidden neurons
    n_o = 2 # outputs
    [w_ih, w_ho] = decode_chromosome(chromosome, n_i, n_h, n_o, w_max)

    # Variable initialization:
    truck.position = 0.0
    truck.velocity = 20
    truck.angle = get_slope_angle(truck.position, slope_index, data_set_index)
    truck.gear = 7
    truck.temperature = 500
    truck.pedal_pressure = 0.0

    angle_evolution = []
    pedal_pressure_evolution = []
    gear_evolution = []
    velocity_evolution = []
    temperature_evolution = []
    position_evolution = []

    angle_evolution.append(truck.angle)
    pedal_pressure_evolution.append(truck.pedal_pressure)
    gear_evolution.append(truck.gear)
    velocity_evolution.append(truck.velocity)
    temperature_evolution.append(truck.temperature)
    position_evolution.append(truck.position)

    
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

    angle_evolution.append(truck.angle)
    pedal_pressure_evolution.append(truck.pedal_pressure)
    gear_evolution.append(truck.gear)
    velocity_evolution.append(truck.velocity)
    temperature_evolution.append(truck.temperature)
    position_evolution.append(truck.position)

    truck.velocity = truck.velocity_change(time_step, truck.velocity)
    if truck.velocity > maximum_velocity:
        fitness = 0
        print('Stopped at x = 0')
        return fitness

    

    time = 0
    while truck.position < road_length:

        velocity_evolution.append(truck.velocity)
        

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
            print(f'Stopped at x = {truck.position}, Fitness = {fitness}')
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

        angle_evolution.append(truck.angle)
        pedal_pressure_evolution.append(truck.pedal_pressure)
        gear_evolution.append(truck.gear)
        temperature_evolution.append(truck.temperature)
        position_evolution.append(truck.position)

        # With all the variables updated, we calculate the velocity that will get the truck to its next point
        truck.velocity = truck.velocity_change(time_step, truck.velocity)

        # If any constraint is violated, we terminate the process
        if (truck.velocity < minimum_velocity) or (truck.velocity > maximum_velocity) or (truck.temperature > maximum_temperature):
            if time != 0:
                average_velocity = truck.position/time
                fitness = truck.position * average_velocity
            if time == 0:
                fitness = 0
            print(f'Stopped at x = {truck.position}, Fitness = {fitness}')
            return angle_evolution, pedal_pressure_evolution, gear_evolution, velocity_evolution, temperature_evolution, position_evolution, fitness
            

        time += time_step

    angle_evolution.append(truck.angle)
    pedal_pressure_evolution.append(truck.pedal_pressure)
    gear_evolution.append(truck.gear)
    temperature_evolution.append(truck.temperature)
    position_evolution.append(truck.position)

    return angle_evolution, pedal_pressure_evolution, gear_evolution, velocity_evolution, temperature_evolution, position_evolution, fitness








# Parameters
data_set_index = 3
slope_index = 3
road_length = 1000
minimum_velocity = 1
maximum_velocity = 25
maximum_angle = 10
maximum_temperature = 750
time_step = 0.1
number_of_hidden_neurons = 4
w_max = 10

angle_evolution, pedal_pressure_evolution, gear_evolution, velocity_evolution, temperature_evolution, position_evolution, fitness = evolution_in_slope(chromosome, data_set_index, slope_index, road_length, minimum_velocity, maximum_velocity, maximum_angle, maximum_temperature, time_step, number_of_hidden_neurons, w_max)




# Plot
label_size = 16
tick_size = 14
legend_size = 14

fig, axs = plt.subplots(2, 3, figsize=(12, 6))
axs = axs.flatten()

fig.delaxes(axs[-1]) # I do not use the last plot since I just have 5

axs[0].plot(position_evolution, angle_evolution, color='blue')
axs[0].set_title(r"Slope angle", fontsize=label_size)
axs[0].set_ylabel(r"$\alpha(x)$", fontsize=label_size)
axs[0].set_xlabel(r"$x$", fontsize=label_size)
axs[0].tick_params(axis='both', labelsize=tick_size)

axs[1].plot(position_evolution, pedal_pressure_evolution, color='blue')
axs[1].set_title(r"Pedal pressure", fontsize=label_size)
axs[1].set_ylabel(r"$P_p(x)$", fontsize=label_size)
axs[1].set_xlabel(r"$x$", fontsize=label_size)
axs[1].tick_params(axis='both', labelsize=tick_size)

axs[2].plot(position_evolution, gear_evolution, color='blue')
axs[2].set_title(r"Gear", fontsize=label_size)
axs[2].set_ylabel(r"$\text{Gear}(x)$", fontsize=label_size)
axs[2].set_xlabel(r"$x$", fontsize=label_size)
axs[2].tick_params(axis='both', labelsize=tick_size)
axs[2].text(0.95, 0.95, f"Fitness: {fitness:.0f}", transform=axs[2].transAxes, ha='right', va='top', fontsize=legend_size, fontweight = 'bold')


axs[3].plot(position_evolution, velocity_evolution, color='black')
axs[3].set_title(r"Velocity", fontsize=label_size)
axs[3].set_ylabel(r"$v(x)$", fontsize=label_size)
axs[3].set_xlabel(r"$x$", fontsize=label_size)
axs[3].tick_params(axis='both', labelsize=tick_size)

axs[4].plot(position_evolution, temperature_evolution, color='black')
axs[4].set_title(r"Temperature", fontsize=label_size)
axs[4].set_ylabel(r"$T(x)$", fontsize=label_size)
axs[4].set_xlabel(r"$x$", fontsize=label_size)
axs[4].tick_params(axis='both', labelsize=tick_size)

plt.tight_layout()
plt.show()


