
import numpy as np
import random
from function_data import load_function_data
from best_chromosome import best_chromosome as chromosome

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sympy as sp
from typing import List, Dict


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


def error_evaluation(x_data, function_data, chromosome, constant_registers, number_of_variable_registers):

    x_data = np.asarray(x_data)
    function_data = np.asarray(function_data)

    chromosome_results = []

    for x_value in x_data:
        chromosome_function_in_x_value = decoding_one_chromosome(x_value, chromosome, number_of_variable_registers, constant_registers)
        chromosome_results.append(chromosome_function_in_x_value)

    chromosome_results = np.array(chromosome_results, dtype = float)
    
    error_sum = np.sqrt(np.mean((function_data - chromosome_results)**2))

    return error_sum, chromosome_results


def chromosome_evaluation(error_sum, chromosome):
    # To measure which chromosome has a better fitness function, we check how accurate the function defined by each chromosome 
    # approximates the data.
    
    if error_sum != 0:
        fitness_value = 1/error_sum
        # Length penalty only for those with length greater than 75
        penalty = max(1, len(chromosome)/75)
        fitness_value = fitness_value/penalty

    else:
        fitness_value = np.inf

    return fitness_value

x_data = []
function_data = []
samples = load_function_data()
for i in range(len(samples)):
    x_data.append(samples[i][0])
    function_data.append(samples[i][1])

constant_registers = [1.0, -1.0, 2.0, 3.0, 4.0]
number_of_variable_registers = 7

error_sum, function_chromosome = error_evaluation(x_data, function_data, chromosome, constant_registers, number_of_variable_registers)
fitness_value = chromosome_evaluation(error_sum, chromosome)


def lgp_to_symbolic_sympy(chromosome: List[List[int]]) -> Dict[str, sp.Expr]:
    
    # For the operands, we first have number_of_variable_registers and then the constat registers
    x = sp.Symbol('x')
    variable_registers_sym = {i: x if i == 1 else sp.Integer(0) for i in range(1,number_of_variable_registers+1)}
    constant_registers_sym = {number_of_variable_registers+1+i: sp.Float(c) for i,c in enumerate(constant_registers)}

    def get_val(idx):
        return variable_registers_sym[idx] if idx <= number_of_variable_registers else constant_registers_sym[idx]

    for instruction_index in range(len(chromosome)):

        operation_index = chromosome[instruction_index][0]
        destination_register_index = chromosome[instruction_index][1]
        operand_1 = chromosome[instruction_index][2]
        operand_2 = chromosome[instruction_index][3]

        working_operand_1 = get_val(operand_1)
        working_operand_2 = get_val(operand_2)
        
        if operation_index == 1:
            variable_registers_sym[destination_register_index] = working_operand_1 + working_operand_2
        elif operation_index == 2:
            variable_registers_sym[destination_register_index] = working_operand_1 - working_operand_2
        elif operation_index == 3:
            variable_registers_sym[destination_register_index] = working_operand_1 * working_operand_2
        elif operation_index == 4:
            if working_operand_2.is_Number and working_operand_2 == 0:
                variable_registers_sym[destination_register_index] = sp.Integer(5000)
            else:
                variable_registers_sym[destination_register_index] = working_operand_1 / working_operand_2

    estimate_expression = sp.simplify(variable_registers_sym[1], rational=True)

    return estimate_expression



# Printing
print(f'Error value: {error_sum} \n')

result = lgp_to_symbolic_sympy(chromosome)
print(f'Chromosome length (number of instructions): {len(chromosome)} \n')

print("Final symbolic expression for the estimation of g(x):")
print(result)


# Plot
label_fontsize = 18
tick_fontsize = 15
legend_fontsize = 13

fig, ax = plt.subplots(figsize=(5, 5))

l1, = ax.plot(x_data, function_data, '.r', label='Data')
l2, = ax.plot(x_data, function_chromosome,'k', label='Estimate', linewidth=1.5)
extra_line = Line2D([], [], color='none')

ax.set_xlabel('x', fontsize=label_fontsize)
ax.set_ylabel('g(x)', fontsize=label_fontsize)
ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
ax.legend(handles=[l1, l2, extra_line], labels=['Data', 'Estimate', f'Error: {error_sum:.3g}'], fontsize=legend_fontsize, loc='lower right')

plt.tight_layout()
plt.show()






