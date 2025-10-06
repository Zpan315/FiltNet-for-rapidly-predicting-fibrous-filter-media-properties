### intergrate the original list into a new list with a specified number of values

import numpy
def intergrate_values(y, NumOfGroups):

    y = [float(x) for x in y] # make the element in a list into floats
    
    num_values = len(y)
    num_groups = NumOfGroups
    values_per_group = num_values // num_groups
    remainder = num_values % num_groups

    # Initialize an empty list for the averaged values
    averaged_values = []

    # Calculate the averages and store them in the averaged_values list
    start = 0
    for _ in range(num_groups):
        end = start + values_per_group
        if remainder > 0:
            end += 1
            remainder -= 1
        group = y[start:end]
        group_average = numpy.mean(group)
        averaged_values.append(group_average)
        start = end

    return(averaged_values)
