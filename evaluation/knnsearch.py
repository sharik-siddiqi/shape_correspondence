import numpy as np
from scipy.spatial.distance import cdist


def knnsearch(A, B):
    """Perform k-nearest neighbor search to find the closest points in B for each point in A.

    Args:
        A (ndarray): An array of shape (n, d) representing n points in d-dimensional space.
        B (ndarray): An array of shape (m, d) representing m points in d-dimensional space.

    Returns:
        ndarray: An array of shape (n,) containing the indices of the closest points in B for each point in A.

    """
    # Compute the pairwise distances between points in A and B
    dist = cdist(A, B)

    # Find the index of the closest point in B for each point in A
    match = np.argmin(dist, axis=1)

    return match
