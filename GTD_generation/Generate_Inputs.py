import time
import random

def generate_Inputs(inputtime, rangeMin, rangeMax,NDecimal):

       # Get the current time in nanoseconds
       current_time_ns = int(time.time() * 1e9) + int(inputtime)

       # Set the random seed to the current time
       random.seed(current_time_ns)
       
       random_number = round(random.uniform(rangeMin, rangeMax),NDecimal)
       
       return(random_number)
