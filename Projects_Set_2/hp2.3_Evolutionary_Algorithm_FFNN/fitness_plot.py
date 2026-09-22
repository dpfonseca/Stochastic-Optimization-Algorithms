
import matplotlib.pyplot as plt

from data_fitness_evolution import training_best_fitness_list
from data_fitness_evolution import validation_best_fitness_list



number_of_generations = len(training_best_fitness_list)
generations = range(1, number_of_generations +1)

label_size = 18
tick_size = 13

fig, axs = plt.subplots(2, 1, figsize=(8, 6))
axs = axs.flatten()

axs[0].plot(generations, training_best_fitness_list, color='blue')
axs[0].set_title(r"Training best fitness", fontsize=label_size)
axs[0].set_ylabel(r"Fitness", fontsize=label_size)
axs[0].set_xlabel(r"Generations", fontsize=label_size)
axs[0].tick_params(axis='both', labelsize=tick_size)

axs[1].plot(generations, validation_best_fitness_list, color='black')
axs[1].set_title(r"Validation best fitness", fontsize=label_size)
axs[1].set_ylabel(r"Fitness", fontsize=label_size)
axs[1].set_xlabel(r"Generations", fontsize=label_size)
axs[1].tick_params(axis='both', labelsize=tick_size)

plt.tight_layout()
plt.show()