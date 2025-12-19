from lab1 import charts_generator
from tests.test_matrix_multiplication import test_matrix_multiplication_fn
from lab1.binet import binet_multiplication
from lab1.strassen import strassen_multiplication
from lab1.AI_algo import *
from tests.test_ai_algo import *

test_matrix_multiplication_fn(binet_multiplication)
# test_matrix_multiplication_fn(strassen_multiplication)
# test_calculate_matrix_C(new_AI)
# charts_generator.generate_chart()
