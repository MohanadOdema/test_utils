import numpy as np
import os
import re

def ret_full_path(dir):
    '''fn to yield the abs path of a given dir if it exists'''
    absolute_path = os.getcwd().split('/')
    base_path = ''
    for name in absolute_path:
        base_path = base_path + str(name) + '/'
        if re.match(dir, name):
            return base_path
    raise RuntimeError("Given directory {} does not appear to be on the current working directory path.".format(dir))

def convert_1d_to_2d_indices(index, n_cols):
    '''fn to convert int index to a 2d index given row_major matrix with 'n_col' columns '''
    return (index // n_cols, index % n_cols)

def convert_2d_to_1d_indices(tuple_index, n_cols):
    '''fn to convert int index to a 2d index given row_major matrix with 'n_col' columns '''
    return tuple_index[0] * n_cols + tuple_index[1]

def reverse_col_index(index, width):
    '''fn to reverse the col index of a tuple'''
    xdim = index[0]
    ydim = (width - 1) - index[-1]
    return (xdim, ydim)

def csr_from_adj(adj):
  """Converts an adjacency matrix to CSR representation.

  Args:
    adj: The adjacency matrix.

  Returns:
    The CSR representation of the adjacency matrix.
  """

  # Get the number of rows and columns in the adjacency matrix.
  n_rows, n_cols = adj.shape

  # Create the CSR row pointer array.
  row_ptr = np.zeros(n_rows + 1, dtype=np.int32)
  for i in range(n_rows):
    row_ptr[i + 1] = row_ptr[i] + np.count_nonzero(adj[i, :])

  # Create the CSR column index array.
  col_ind = np.zeros(np.count_nonzero(adj), dtype=np.int32)
  k = 0
  for i in range(n_rows):
    for j in range(n_cols):
      if adj[i, j] != 0:
        col_ind[k] = j
        k += 1

  # Create the CSR value array.
  val = np.zeros(np.count_nonzero(adj), dtype=np.float32)
  k = 0
  for i in range(n_rows):
    for j in range(n_cols):
      if adj[i, j] != 0:
        val[k] = adj[i, j]
        k += 1

  # Return the CSR representation of the adjacency matrix.
  return row_ptr, col_ind, val

def generate_matrix(n, m):
  """Generates a matrix of 0s and 1s.

  Args:
    n: The number of rows in the matrix.
    m: The number of columns in the matrix.

  Returns:
    The matrix of 0s and 1s.
  """

  # Create a matrix of zeros.
  matrix = np.zeros((n, m))
  # Randomly generate 1s in the matrix.
  for i in range(n):
    for j in range(m):
      if np.random.random() < 0.5:
        matrix[i, j] = 1

  return matrix


if __name__ == '__main__':
   
    # e.g. CSR
    adj = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 1., 0., 1., 0.],
       [1., 0., 0., 0., 0., 1., 1., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0.]])

    print(csr_from_adj(adj))    
