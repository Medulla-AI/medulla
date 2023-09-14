import unittest
import numpy as np
from nlp import * 

class TestEmbeddingsCompare(unittest.TestCase):
    def test_cosine_similarity(self):
        # Test case 1: Check cosine similarity with identical vectors
        source = np.array([1.0, 2.0, 3.0])
        candidates = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        expected_result = np.array([1.0, 0.97463185, 0.95941195])
        result = embeddings_compare(source, candidates)
        print(result)
        np.testing.assert_allclose(result, expected_result, rtol=1e-3)

    def test_cosine_similarity_with_zeros(self):
        # Test case 2: Check cosine similarity with zero vectors
        source = np.array([0.0, 0.0])
        candidates = np.array([[1.0, 0.0], [0.0, 1.0]])
        expected_result = np.array([0.0, 0.0])
        result = embeddings_compare(source, candidates)
        print(result)
        np.testing.assert_allclose(result, expected_result, rtol=1e-3)

    def test_cosine_similarity_with_opposite_vectors(self):
        # Test case 3: Check cosine similarity with opposite vectors
        source = np.array([1.0, 0.0])
        candidates = np.array([[-1.0, 0.0], [0.0, -1.0]])
        expected_result = np.array([-1.0, 0.0])
        result = embeddings_compare(source, candidates)
        np.testing.assert_allclose(result, expected_result, rtol=1e-3)




if __name__ == '__main__':
    unittest.main()