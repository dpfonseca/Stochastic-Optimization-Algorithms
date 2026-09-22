import math

def rescaling(original_value, original_minimum, original_maximum, new_minimum, new_maximum):

    new_value = new_minimum + (original_value - original_minimum) * (new_maximum - new_minimum)/(original_maximum - original_minimum)

    return new_value




# Test of encoding and decoding
#
# Encoding: Starting from two weight matrices (W^IH and W^HO, denoted w_ih and w_ho in
#           the Python code), of size Nh x (Ni+1) and No x (Nh+1), respectively, 
#           generate a single chromosome, such that the first Nh x (Ni+1) genes are
#           used for obtaining W^IH and the remaining genes for W^HO. 
#           The genes should take values in the range [0,1] 
#           (e.g. 0.219412987333, 0.452951914512, and so on)
#
#
#######################################################################
#  ToDo: Write the encode_network function:
#  Do NOT change the interface!
#######################################################################

def encode_network(w_ih,w_ho,w_max):

    w_ih_number_of_rows = len(w_ih)
    w_ih_number_of_columns = len(w_ih[0])

    w_ho_number_of_rows = len(w_ho)   
    w_ho_number_of_columns = len(w_ho[0])

    chromosome_part_1 = [] 
    for row_index in range(w_ih_number_of_rows):
        for column_index in range(w_ih_number_of_columns):
            old_value = w_ih[row_index][column_index]
            new_value = rescaling(old_value, -w_max, w_max, 0, 1)
            chromosome_part_1.append(new_value)

    chromosome_part_2 = []
    for row_index in range(w_ho_number_of_rows):
        for column_index in range(w_ho_number_of_columns):
            old_value = w_ho[row_index][column_index]
            new_value = rescaling(old_value, -w_max, w_max, 0, 1)
            chromosome_part_2.append(new_value)

    chromosome = chromosome_part_1 + chromosome_part_2
    
    return chromosome

 # Add code here!

#
# Decoding: Starting from a chromosome of length Nh x (Ni+1) + No x (Nh+1), use
#           The first Nh x (Ni+1) genes to obtain W^IH, and the remaining genes
#           to obtain W^OH. The weights and the biases (i.e., the elements of the matrices) 
#           should take values in the range [-wMax,wMax], so the alleles (values of the
#           genes in the chromosome) must be rescaled correctly.
#
#
#######################################################################
#  ToDo: Write the decode_chromosome function:
#  Do NOT change the interface!
#######################################################################

def decode_chromosome(chromosome, n_i, n_h, n_o, w_max):

    w_ih = []
    w_ho = []

    w_ih_number_of_rows = n_h
    w_ih_number_of_columns = n_i + 1

    starting_point_for_w_ho = w_ih_number_of_rows * w_ih_number_of_columns
    w_ho_number_of_rows = n_o
    w_ho_number_of_columns = n_h + 1

    for row_index in range(0, w_ih_number_of_rows):
        new_ih_row = []
        for column_index in range(0,w_ih_number_of_columns):
            starting_element_row = row_index * w_ih_number_of_columns
            addition_for_column = column_index
            old_value = chromosome[starting_element_row + addition_for_column]
            new_value = rescaling(old_value, 0, 1, -w_max, w_max)
            new_ih_row.append(new_value)

        w_ih.append(new_ih_row)

    for row_index in range(0, w_ho_number_of_rows):
        new_ho_row = []
        for column_index in range(0, w_ho_number_of_columns):
            starting_element_row = starting_point_for_w_ho + (row_index * w_ho_number_of_columns)
            addition_for_column = column_index
            old_value = chromosome[starting_element_row + addition_for_column]
            new_value = rescaling(old_value, 0, 1, -w_max, w_max)
            new_ho_row.append(new_value)

        w_ho.append(new_ho_row)

    return w_ih, w_ho


  
  



############################################################
# Main program:
############################################################

if __name__ == "__main__":

  # The maximum (absolute) value of weights and biases. Thus, they take values in
  # the range [-w_max,w_max]
  w_max = 5

  # Sample network of size 3-3-2. Note the the number of rows in w_ih MUST be
  # equal to the number of columns in w_ho, minus 1; see also the definition of nH below.
  #
  # Note: Your encoding and decoding methods should work for any values of nI, nH, and nO,
  # not just for the example below! Thus, test your encoding and decoding functions by
  # defining different set of matrices w_ih and w_ho (fulfilling the criterion on nH, see below)
  #
  w_ih = [ [2, 1, -3, 1], [5, -2, 1, 4], [3, 0, 1, 2]]
  w_ho = [[1, 0, -4, 3], [4, -2, 0, 1]]
  n_i = len(w_ih[0])-1
  n_h = len(w_ih) # % must be equal to len(w_ho[0])-1, for a valid set of matrices for an FFNN
  n_o = len(w_ho)

  chromosome = encode_network(w_ih,w_ho,w_max)
  [new_w_ih, new_w_ho] = decode_chromosome(chromosome,n_i,n_h,n_o,w_max)

  error_count = 0
  tolerance = 0.00000001
  for i in range(n_h):
    for j in range(n_i+1):
      difference = abs(w_ih[i][j]-new_w_ih[i][j])
      if (difference > tolerance):
        print("Error for element " + str(i) + " , " + str(j) + " in wIH")
        error_count += 1

  for i in range(n_o):
    for j in range(n_h+1):
      difference = abs(w_ho[i][j]-new_w_ho[i][j])
      if (difference > tolerance):
        print("Error for element " + str(i) + " , " + str(j) + " in wHO")
        error_count += 1

  if (error_count == 0):
    print("Test OK")
  else:
    print("Test failed")
  input(f'Press return to exit')
