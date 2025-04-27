import warnings

import igl
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")


def data_augmentation(point_set):
    """Apply random rotation to the point cloud for data augmentation."""
    theta = np.random.uniform(-np.pi / 2, np.pi / 2)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])
    point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(
        rotation_matrix
    )  # Random rotation
    return point_set


def pc_normalize(pc):
    """Normalize the point cloud to have zero mean."""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    return pc


def farthest_point_sample(point, npoint, match):
    """Perform farthest point sampling on the point cloud.

    Args:
        point (ndarray): Input point cloud of shape (N, D).
        npoint (int): Number of points to sample.
        match (ndarray): Matching indices for downsampling.

    Returns:
        tuple: Sampled point cloud and sampled indices.

    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    centroids[:npoint] = match
    point = point[centroids.astype(np.int32), :]
    return point, centroids.astype(np.int32)


class Surr12kModelNetDataLoader(Dataset):
    """DataLoader for loading and processing 3D shape datasets."""

    def __init__(
        self,
        root,
        npoint=1024,
        split="train",
        uniform=False,
        augm=False,
    ):
        """Initialize the DataLoader.

        Args:
            root (str): Root directory of the dataset.
            npoint (int): Number of points to sample.
            split (str): Dataset split ('train' or 'test').
            uniform (bool): Whether to use uniform sampling.
            augm (bool): Whether to apply data augmentation.

        """
        self.uniform = uniform
        self.augm = augm
        self.npoints = npoint
        self.split = split

        # Load the simplified mesh
        _, self.f = igl.read_triangle_mesh(
            "<path_to_simplified_mesh.obj>"
        )  # Placeholder for simplified mesh file

        # Load matching coordinates for downsampling (2100 -> 1000)
        match_fixed = loadmat(
            "<path_to_match_2100_1000.mat>", variable_names=["match"]
        )  # Placeholder for match file
        self.match = match_fixed["match"][:, 0] - 1

        # Load dataset
        self.data = self._load_dataset(root, split)

    def _load_dataset(self, root, split):
        """Load the dataset based on the split.

        Args:
            root (str): Root directory of the dataset.
            split (str): Dataset split ('train' or 'test').

        Returns:
            ndarray: Loaded dataset.

        """
        if split == "train":
            dataset_path = "<path_to_training_data.mat>"  # Placeholder for training data
        else:
            dataset_path = "<path_to_testing_data.mat>"  # Placeholder for testing data

        data = loadmat(dataset_path, variable_names=["vert"])["vert"]
        return data

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, index):
        """Get a single item from the dataset.

        Args:
            index (int): Index of the item.

        Returns:
            tuple: Processed point cloud and geodesic distances.

        """
        point_set = self.data[index, :, :]

        # Perform uniform sampling if required
        if self.uniform:
            point_set, indices = farthest_point_sample(
                point_set, self.npoints, self.match
            )
            area = igl.massmatrix(
                point_set,
                self.f,
                igl.MASSMATRIX_TYPE_VORONOI,
            ).toarray()
            area = area / area.sum()
        else:
            indices = np.arange(self.npoints)
            area = np.ones(self.npoints) / self.npoints

        # Apply data augmentation if required
        if self.augm:
            point_set = data_augmentation(point_set)

        # Normalize the point cloud
        point_set = pc_normalize(point_set)

        return (
            point_set.astype(np.float32),
            indices.astype(np.int64),
            area.astype(np.float32),
        )
