from lab2 import charts_generator
from lab2 import data_generators
from lab2.LU import lu_recursive
from lab2.Gauss import recursive_gauss_elimination

from tests.test_matrix_inversion import test_matrix_inversion_fn
from tests.test_LU import test_lu_fn
from tests.test_gauss_fn import test_gauss_fn

from lab1.binet import binet_multiplication
from lab1.strassen import strassen_multiplication
from lab2.inversion import recursion_inversion

# test_matrix_inversion_fn(recursion_inversion, binet_multiplication)
# test_matrix_inversion_fn(recursion_inversion,strassen_multiplication)

data_generators.generate_inv(method="Binet")
data_generators.generate_inv(method="Strassen")
charts_generator.generate_inv()


# test_lu_fn(lu_recursive,recursion_inversion,binet_multiplication)
# test_lu_fn(lu_recursive,recursion_inversion,strassen_multiplication)

# data_generators.generate_lu(method="Binet")
# data_generators.generate_lu(method="Strassen")
# charts_generator.generate_lu()




#$ test_gauss_fn(recursive_gauss_elimination,recursion_inversion, binet_multiplication,lu_recursive)
# test_gauss_fn(recursive_gauss_elimination,recursion_inversion,strassen_multiplication,lu_recursive)

#data_generators.generate_gauss(method="Binet")
#data_generators.generate_gauss(method="Strassen")
# charts_generator.generate_gauss()
