import numpy as np


def compute_geodesic_error(correspondences, geodesic_distances):
    """Compute the geodesic error for given correspondences and geodesic distances.

    Args:
        correspondences (list): List of correspondences between points.
        geodesic_distances (ndarray): Geodesic distance matrix.

    Returns:
        ndarray: Normalized geodesic errors.

    """
    # Initialize the error matrix
    errors = np.zeros((
        geodesic_distances.shape[0],
        len(correspondences),
    ))

    # Compute the diameter of the shape (maximum geodesic distance)
    diameter = np.max(geodesic_distances)

    # Calculate geodesic errors for each correspondence
    for idx, corr in enumerate(correspondences):
        for j in range(errors.shape[0]):
            errors[j, idx] = (
                geodesic_distances[j, corr[j]]
                + geodesic_distances[corr[j], j]
            ) / 2

    # Normalize errors by the diameter
    return errors / diameter


def compute_error_curves(errors, thresholds):
    """Compute error curves for all correspondences based on thresholds.

    Args:
        errors (ndarray): Geodesic errors for all correspondences.
        thresholds (ndarray): Threshold values for error computation.

    Returns:
        ndarray: Error curves for all correspondences.

    """
    # Initialize the error curves matrix
    curves = np.zeros((errors.shape[1], thresholds.shape[0]))

    # Compute the error curve for each correspondence
    for i in range(errors.shape[1]):
        curves[i, :] = compute_single_error_curve(
            errors[:, i],
            thresholds,
        )

    return curves
