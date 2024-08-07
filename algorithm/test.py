import numpy as np

coordinates = np.array([0, 0, 4, 5, 18, 14, 5, 7, 37, 28, 5, 6, 0, 42, 4, 4, 18, 0, 6, 6, 26, 14, 4, 5, 0, 28, 7, 10, 18, 42, 5, 7, 37, 0, 5, 6, 0, 14, 5, 5, 18, 28, 5, 5, 37, 42, 4, 6])

# Number of rows (each set of 4 coordinates will form a row)
num_rows = len(coordinates) // 4

# Initialize the matrix
matrix = np.zeros((num_rows, 4) , dtype=int)

for i, y in enumerate(np.arange(num_rows)):
    matrix[i, 0] = coordinates[4 * i + 0]
    matrix[i, 1] = coordinates[4 * i + 1]
    matrix[i, 2] = coordinates[4 * i + 0] + coordinates[4 * i + 2]
    matrix[i, 3] = coordinates[4 * i + 1] + coordinates[4 * i + 3]

### position of facilities.
### y_from, x_from, y_to, x_to 
print(matrix)