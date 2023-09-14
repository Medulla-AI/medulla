import unittest
import numpy as np
from metrics import * 

# class TestPearsonCorrelation(unittest.TestCase):

#     def test_pearson_correlation_batch(self):
#         # Test case 4: Check Pearson correlation with vectors of different dimensions
#         x = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
#         y = np.array([[2.0, 1.0, 6.0], [1.0, 2.0, 3.0]])  
#         expected_result = np.array([0.7559289460184544, 1.0])
#         result = pearsonr(x, y)
#         np.testing.assert_allclose(result, expected_result, rtol=1e-5)

#     def test_pearson_correlation_single(self):
#         # Test case 4: Check Pearson correlation with vectors of different dimensions
#         x = np.array([1.0, 2.0, 3.0])
#         y = np.array([2.0, 1.0, 6.0])  
#         expected_result = np.array(0.7559289460184544)
#         result = pearsonr(x, y)
#         np.testing.assert_allclose(result, expected_result, rtol=1e-5)

#     def test_pearson_correlation_with_identical_vectors(self):
#         # Test case 1: Check Pearson correlation with identical vectors
#         x = np.array([1.0, 2.0, 3.0])
#         y = np.array([1.0, 2.0, 3.0])
#         expected_result = 1.0
#         result = pearsonr(x, y)
#         self.assertAlmostEqual(result, expected_result, places=5)

#     def test_pearson_correlation_with_opposite_vectors(self):
#         # Test case 2: Check Pearson correlation with opposite vectors
#         x = np.array([1.0, 2.0, 3.0])
#         y = np.array([-1.0, -2.0, -3.0])
#         expected_result = -1.0
#         result = pearsonr(x, y)
#         self.assertAlmostEqual(result, expected_result, places=5)

if __name__ == '__main__':
    unittest.main()