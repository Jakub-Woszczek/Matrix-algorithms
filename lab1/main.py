from binet import *

A = [[1, 2, 3], [4, 5, 6]]

B = [[7, 8], [9, 10], [11, 12]]

# Expected result
C = [[58, 64], [139, 154]]

c = binet_multiplication(A, B)
for row in c:
    print(row)
