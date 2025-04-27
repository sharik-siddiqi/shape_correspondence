import numpy as np
from knnsearch import knnsearch


def our_match(phiM, phiN):
    """Compute point-to-point correspondences between two shapes using basis functions.

    Args:
        phiM (ndarray): Basis functions for the source shape, of shape (n, k).
        phiN (ndarray): Basis functions for the target shape, of shape (m, k).

    Returns:
        ndarray: An array of shape (n,) containing the indices of the closest points in phiN for each point in phiM.

    """
    # Compute the transformation matrix C using the pseudoinverse of phiM
    C = np.linalg.pinv(phiM) @ phiN

    # Find the closest points in phiN for each point in phiM after applying the transformation
    match = knnsearch(phiM @ C, phiN)

    return match


def our_match_desc(phiM, phiN, descM, descN):
    """Compute point-to-point correspondences between two shapes using both basis functions and descriptors.

    Args:
        phiM (ndarray): Basis functions for the source shape, of shape (n, k).
        phiN (ndarray): Basis functions for the target shape, of shape (m, k).
        descM (ndarray): Descriptors for the source shape, of shape (n, d).
        descN (ndarray): Descriptors for the target shape, of shape (m, d).

    Returns:
        tuple: A tuple containing:
            - match_1 (ndarray): Indices of the closest points in phiN for each point in phiM.
            - transformed_phiM (ndarray): Transformed basis functions for the source shape.
            - phiN (ndarray): Basis functions for the target shape (unchanged).
            - match_2 (ndarray): Indices of the closest points in descN for each point in descM.

    """
    # Compute the transformation matrices F and G using the pseudoinverse of phiM and phiN
    F = np.linalg.pinv(phiM) @ descM
    G = np.linalg.pinv(phiN) @ descN

    # Compute the transformation matrix C using F and the pseudoinverse of G
    C = F @ np.linalg.pinv(G)

    # Find the closest points in phiN for each point in phiM after applying the transformation
    match_1 = knnsearch(phiM @ C, phiN)

    # Find the closest points in descN for each point in descM
    match_2 = knnsearch(descM, descN)

    return match_1, phiM @ C, phiN, match_2
