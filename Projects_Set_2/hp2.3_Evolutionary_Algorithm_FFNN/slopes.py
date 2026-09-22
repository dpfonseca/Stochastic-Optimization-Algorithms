import math
import random

seed = random.randint(0,10000)
seed = 1685
print(f'Slopes -> random seed: {seed}')
random.seed(seed)

# This file provides the FORMAT you should use for the
# slopes in HP2.3. x denotes the horizontal distance
# travelled (by the truck) on a given slope, and
# alpha measures the slope angle at distance x
#
# iSlope denotes the slope index (i.e. 1,2,..10 for the
# training set etc.)
# iDataSet determines whether the slope under consideration
# belongs to the training set (data_set_index = 1), validation
# set (data_set_index = 2), or the test set (data_set_index = 3).
#
# Note that the slopes given below are just EXAMPLES.
# Please feel free to implement your own slopes below,
# as long as they fulfil the criteria given in HP2.3.
#
# You may remove the comments above and below, as they
# (or at least some of them) violate the coding standard 
#  a bit. :)
# The comments have been added as a clarification of the 
# problem that should be solved!).



def make_random_params():
    a = random.randint(4, 6)

    random_1 = random.random()
    if random_1 > 0.5:
        b = random.uniform(0.5, 2)
    else:
        b = -random.uniform(0.5, 2)

    random_2 = random.random()
    if random_2 > 0.5:
        c = random.uniform(0.5,2)
    else:
        c = -random.uniform(0.5,2)

    d = math.sqrt(random.randint(1, 8))
    e = math.sqrt(random.randint(1, 8))
    f = random.choice([50, 60, 70, 80, 90, 100, 200, 500, 1000])
    g = random.choice([50, 60, 70, 80, 90, 100, 200, 500, 1000])
    return (a, b, c, d, e, f, g)

training_parameters = [make_random_params() for i in range(10)]
validation_parameters = [make_random_params() for i in range(5)]
test_parameters = [make_random_params() for i in range(5)]

fraction_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 1000]


def get_slope_angle(x, slope_index, data_set_index):   
    if (data_set_index == 1): # Training
        if (slope_index == 1):
            alpha = 4 + math.sin(x/100) + math.cos(math.sqrt(2)*x/50)
        elif (slope_index == 10):
            alpha = 3 - 2*math.sin(x/50) + math.cos(math.sqrt(2)*x/100)
        else:
            a, b, c, d, e, f, g = training_parameters[slope_index-1]
            alpha = a + b*math.sin(d*x/f) +c*math.cos(e*x/g)


    elif (data_set_index == 2): # Validation
        if (slope_index == 1):
            alpha = 6 - math.sin(x/100) + 0.8*math.cos(math.sqrt(3)*x/50)
        elif (slope_index == 5):
            alpha = 5 + math.sin(math.sqrt(2)*x/50) - math.cos(math.sqrt(8)*x/50)
        else:
            a, b, c, d, e, f, g = validation_parameters[slope_index-1]
            alpha = a + b*math.sin(d*x/f) +c*math.cos(e*x/g)


    elif (data_set_index == 3): # Test
        if (slope_index == 1):
            alpha = 6 - math.sin(x/100) + math.cos(math.sqrt(7)*x/50)
        elif (slope_index == 5):
            alpha = 4 + (x/1000) + math.sin(x/70) + math.cos(math.sqrt(7)*x/100)
        else:
            a, b, c, d, e, f, g = test_parameters[slope_index-1]
            alpha = a + b*math.sin(d*x/f) +c*math.cos(e*x/g)

    return alpha


