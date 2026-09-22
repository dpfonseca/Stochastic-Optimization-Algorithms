# Daniel Peña Fonseca

import numpy as np
import random
import matplotlib.pyplot as plt

seed = 9771
print(f'Random seed: {seed}')
random.seed(seed)
np.random.seed(seed)

def f_function(x_array):
    x_1 = x_array[0]
    x_2 = x_array[1]
    result = (x_1**2 + x_2 -11)**2 + (x_1 + x_2**2 -7)**2
    return result


def log_function(a, x_array):
    result = np.log(a + f_function(x_array))
    return result


def pso(number_of_particles, dimensions, minimum_position_value, maximum_position_value, scaling_factor, time_step_length, function_used, final_time, initial_inertia_weight, C_1_cognitive_constant, C_2_social_constant, maximum_velocity_value, threshold, time_index):
    
    # Initialization
    position_vectors = np.zeros((number_of_particles, dimensions))
    particle_best_position = []

    velocity_vectors = np.zeros((number_of_particles, dimensions))

    function_values_in_current_step = np.zeros(number_of_particles)
    particle_best_function_values = []
    swarm_best_function_values = []

    inertia_weight = initial_inertia_weight
    minimum_inertia_weight = 0.3
    decaying_factor_of_inertia_weight = 0.99

    window_size_of_time = 10

    # Initializing
    for particle_index in range(number_of_particles):
        for dimension_index in range(dimensions):

            r1 = random.random()
            position_vector = minimum_position_value + r1 * (maximum_position_value - minimum_position_value)
            position_vectors[particle_index][dimension_index] = position_vector

            r2 = random.random()
            velocity_vector = (scaling_factor/time_step_length) * ((-0.5 * (maximum_position_value - minimum_position_value)) + r2 * (maximum_position_value - minimum_position_value))
            velocity_vectors[particle_index][dimension_index] = velocity_vector

        
    # For each particle, we will store its best position and fitness
    particle_best_position = position_vectors.copy()
    particle_best_function_values = np.array([function_used(particle_best_position[i]) for i in range(number_of_particles)])
    
    
    # Evaluation
    # Before entering the time loop, we initialize for time = 0
    for particle_index in range(number_of_particles):
        new_function_value = function_used(position_vectors[particle_index][:])
        function_values_in_current_step[particle_index] = new_function_value

    minimum_value_of_function = min(function_values_in_current_step)
    best_index = np.argmin(function_values_in_current_step)

    swarm_best_position = position_vectors[best_index].copy()
    swarm_best_function_values.append(minimum_value_of_function)

    # Updating velocities for the first step
    for particle_index in range(number_of_particles):
            r, q = random.random(), random.random()
            new_velocity_vector = inertia_weight * velocity_vectors[particle_index] + (C_1_cognitive_constant * r/time_step_length) * (particle_best_position[particle_index] - position_vectors[particle_index][:]) + (C_2_social_constant * q/time_step_length) * (swarm_best_position - position_vectors[particle_index][:])
            
            for dimension_index in range(dimensions):
                if new_velocity_vector[dimension_index] > maximum_velocity_value:
                    new_velocity_vector[dimension_index] = maximum_velocity_value

                elif new_velocity_vector[dimension_index] < -maximum_velocity_value:
                    new_velocity_vector[dimension_index] = -maximum_velocity_value

            velocity_vectors[particle_index] = new_velocity_vector
            position_vectors[particle_index] += new_velocity_vector * time_step

    inertia_weight *= decaying_factor_of_inertia_weight


    # Loop for greater times
    for time_index in range(1, final_time):
        
        # If there has been few change during the last time steps, we assume it has converged.
        if time_index > window_size_of_time:
            recent_function_values = swarm_best_function_values[- window_size_of_time:]
            if max(recent_function_values) - min(recent_function_values) < threshold:
                break

        # Updating the function values
        for particle_index in range(number_of_particles):
            new_function_value = function_used(position_vectors[particle_index][:])
            function_values_in_current_step[particle_index] = new_function_value

            if function_values_in_current_step[particle_index] < particle_best_function_values[particle_index]:
                particle_best_position[particle_index] = position_vectors[particle_index].copy()
                particle_best_function_values[particle_index] = function_values_in_current_step[particle_index]

        minimum_value_of_function = min(function_values_in_current_step)
        swarm_best_function_values.append(minimum_value_of_function)
        best_index = np.argmin(function_values_in_current_step)
        swarm_best_position = position_vectors[best_index].copy()


        # Updating
        for particle_index in range(number_of_particles):
                r, q = random.random(), random.random()
                new_velocity_vector = inertia_weight * velocity_vectors[particle_index] + (C_1_cognitive_constant * r/time_step_length) * (particle_best_position[particle_index] - position_vectors[particle_index][:]) + (C_2_social_constant * q/time_step_length) * (swarm_best_position - position_vectors[particle_index][:])
                
                for dimension_index in range(dimensions):
                    if new_velocity_vector[dimension_index] > maximum_velocity_value:
                        new_velocity_vector[dimension_index] = maximum_velocity_value

                    if new_velocity_vector[dimension_index] < -maximum_velocity_value:
                        new_velocity_vector[dimension_index] = -maximum_velocity_value

                velocity_vectors[particle_index] = new_velocity_vector
                position_vectors[particle_index] += new_velocity_vector * time_step

        inertia_weight = max(minimum_inertia_weight, inertia_weight * decaying_factor_of_inertia_weight)


    return swarm_best_position, swarm_best_function_values[-1] 



number_of_particles = 30
dimensions = 2
minimum_position = -5
maximum_position = 5
scaling_factor = 1
time_step = 0.1
time_step_length = 1

final_time = 500
initial_inertia_weight = 1.4
C_1_cognitive_constant = 2
C_2_social_constant = 2
maximum_velocity = 5
threshold = 10**(-12)

a_parameter_for_logarithmic_function = 0.01

# To calculate all the minima I assume that there is a distance greater than tolerance between them
search_iterations = 200
tolerance = 1e-2
all_minimum_points_found = []
all_minimum_function_values_found = []
swarm_best_position, all_time_minimum_function_value = pso(number_of_particles, dimensions, minimum_position, maximum_position, scaling_factor, time_step_length, f_function, final_time, initial_inertia_weight, C_1_cognitive_constant, C_2_social_constant, maximum_velocity, threshold, time_step)
all_minimum_points_found.append(swarm_best_position)
all_minimum_function_values_found.append(all_time_minimum_function_value)

for i in range(search_iterations):
    l=0
    swarm_best_position, all_time_minimum_function_value = pso(number_of_particles, dimensions, minimum_position, maximum_position, scaling_factor, time_step_length, f_function, final_time, initial_inertia_weight, C_1_cognitive_constant, C_2_social_constant, maximum_velocity, threshold, time_step)
    for j in range(len(all_minimum_points_found)):
        if np.linalg.norm(swarm_best_position - np.array(all_minimum_points_found[j])) > tolerance:
            l+=1
    if l == len(all_minimum_points_found):
        all_minimum_points_found.append(swarm_best_position)
        all_minimum_function_values_found.append(all_time_minimum_function_value)

for k in range(len(all_minimum_points_found)):
    print(f'Minimum number {k+1}: x_1 = {all_minimum_points_found[k][0]}, x_2 = {all_minimum_points_found[k][1]}, f(x_1, x_2) = {all_minimum_function_values_found[k]}')



# Plot
label_size = 20     
tick_size = 18        

# I create a grid for plotting
x = np.linspace(minimum_position, maximum_position, 400)
y = np.linspace(minimum_position, maximum_position, 400)
X, Y = np.meshgrid(x, y)

Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = log_function(a_parameter_for_logarithmic_function, [X[i,j], Y[i,j]])

plt.figure(figsize=(8,8))
contour = plt.contour(X, Y, Z, levels=50, cmap='viridis')
plt.clabel(contour, fontsize=12)



for idx, minimum in enumerate(all_minimum_points_found):
    plt.plot(minimum[0], minimum[1], 'kx', markersize=8, markeredgewidth=2)
    

plt.xlabel('X', fontsize=label_size)
plt.ylabel('Y', fontsize=label_size)
plt.xticks(fontsize=tick_size)
plt.yticks(fontsize=tick_size)

plt.tight_layout()
plt.show()



