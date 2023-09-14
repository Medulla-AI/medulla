import numpy as np

# def _structure(array):
#     array = np.asarray(array).T
#     return np.reshape(np.tile(array, 2), (2, len(array)))

# def pearsonr(x: np.ndarray, 
#              y: np.ndarray) -> np.ndarray:
#     """
#     Calculate batch Pearson correlation coefficient between two sets of vectors.
    
#     Args:
#         x (np.ndarray): First set of vectors. (dim) or (n, dim)
#         y (np.ndarray): Second set of vectors. (dim) or (n, dim)
        
#     Returns:
#         np.ndarray: Array of Pearson correlation coefficients.
#     """
#     x = np.asarray(x).T
#     y = np.asarray(y).T

#     assert x.shape == y.shape

#     is_single = False
#     if len(x.shape) == 1:
#         is_single = True
#         x = np.reshape(np.tile(x, 2), (2, len(x)))
#         y = np.reshape(np.tile(y, 2), (2, len(y)))
    
#     x = x - np.expand_dims(np.mean(x, axis=1), axis=-1)
#     y = y - np.expand_dims(np.mean(y, axis=1), axis=-1)
#     sum_of_squares_x = np.einsum('ij,ij -> i', x, x)
#     sum_of_squares_y = np.einsum('ij,ij -> i', y, y)
    
#     pearson_coefficients = np.sum(x * y, axis=-1) / np.sqrt(sum_of_squares_x * sum_of_squares_y)

#     if is_single == True:
#         return pearson_coefficients[0]
    
#     return pearson_coefficients

# x = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 3.0]]) * 0.2
# y = np.array([[2.0, 1.0, 6.0], [1.0, 4.0, 5.0]]) * 0.3
# print(pearsonr(x, y))