#!/usr/bin/python3

import argparse
from typing import Iterable
from typing import Tuple
from math import sqrt

parser = argparse.ArgumentParser()

parser.add_argument("value", help="Calculate and print the median of input values and it's standard deviation", type=float, nargs="+")

args = parser.parse_args()

def calculate_arit_median(values: Iterable[float]) -> Tuple[float, float]:
    if not values:
        return 0,0
    len_values = len(values)
    expected_value = sum(values) / len_values 
    square_sum = 0

    for value in values:
        square_sum = (value - expected_value) ** 2

    variance = square_sum / (len_values - 1)
    standard_deviation = sqrt(variance)

    return expected_value, standard_deviation

arit_median, standard_deviation = calculate_arit_median(args.value)
print(arit_median, standard_deviation)



