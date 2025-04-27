# Import necessary libraries
import os
import random
from code.dataloader import Surr12kModelNetDataLoader as DataLoader

import igl
import numpy as np
import torch
import torch.nn.functional as F
from model import PointNetBasis
from torch import optim
from tqdm import tqdm

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
manualSeed = 1  # fix seed
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Path to a sample mesh in the training dataset
simple_mesh_path = "path/to/your/sample_mesh.mat"

# Obtain the faces of one of the shapes in the dataset
_, f = igl.read_triangle_mesh(simple_mesh_path)

# Batch size for training and validation
b_size = 10

# Create output directory for saving trained models
OUT_DIR = "./models/trained"
os.makedirs(OUT_DIR, exist_ok=True)

# Path to training dataset
DATA_PATH = "path/to/your/train_dataset"

# -----------------------------------------------------------------------------------
# Loading SHOT, HKS, and WKS descriptor data
#
# IMPORTANT: {PREPARE ACCORDINGLY FOR YOUR OWN DATA}
#
# - Load SHOT, Heat Kernel Signature (HKS), and Wave Kernel Signature (WKS) descriptor files for the meshes in the dataset.
# - If using a large dataset, split descriptors into multiple files and concatenate.
#
# Example:
# shot_0 = loadmat('./shot_surreal.mat')  # SHOT descriptors
# hks_0  = loadmat('./hks_surreal.mat')  # HKS descriptors
# wks_0  = loadmat('./wks_surreal.mat')  # WKS descriptors
#
# Prepare training and testing splits:
# - shot_train -> concatenate all subdivided SHOT descriptor files for training
# - hks_train  -> concatenate all subdivided HKS descriptor files for training
# - wks_train  -> concatenate all subdivided WKS descriptor files for training
#
# - shot_test  -> concatenate all subdivided SHOT descriptor files for testing
# - hks_test   -> concatenate all subdivided HKS descriptor files for testing
# - wks_test   -> concatenate all subdivided WKS descriptor files for testing
# -----------------------------------------------------------------------------------

# For our calculations, we selected a combination of SHOT and WKS descriptors for our calculations
desc_train = np.concatenate((shot_train, wks_train), axis=2)
desc_test = np.concatenate((shot_test, wks_test), axis=2)

# Custom dataloader instance
TRAIN_DATASET = DataLoader(
    root=DATA_PATH,
    npoint=1000,
    split="train",
    uniform=True,
    normal_channel=False,
    augm=True,
    rand=False,
)
TEST_DATASET = DataLoader(
    root=DATA_PATH,
    npoint=1000,
    split="test",
    uniform=True,
    normal_channel=False,
    augm=True,
    rand=False,
)

train_loader = torch.utils.data.DataLoader(
    TRAIN_DATASET,
    batch_size=b_size,
    shuffle=True,
    num_workers=0,
)
test_loader = torch.utils.data.DataLoader(
    TEST_DATASET,
    batch_size=b_size,
    shuffle=True,
    num_workers=0,
)

# Initialise PointNet Basis Network with 20 basis
basisNet = PointNetBasis(k=20, feature_transform=False)
basisNet.to(device)

# Define optimizer and learning rate scheduler
optimizer = optim.Adam(
    basisNet.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
)
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=50,
    gamma=0.7,
)

# Initialize variables to track the best evaluation loss
best_eval_loss = np.inf

# Generate identity matrix for loss calculation
iden_1 = np.identity(20).reshape(1, 20, 20)
iden_1 = torch.Tensor(np.repeat(iden_1, b_size, axis=0)).to(device)


# Descriptors loss
def desc_loss(phi_A, phi_B, G_A, G_B, area):
    """Calculate the descriptor loss between two shapes.

    Args:
        phi_A, phi_B: Basis functions for shapes A and B.
        G_A, G_B: Descriptors for shapes A and B.
        area: Area tensor for normalization.

    Returns:
        Q: Normalized projection matrix.
        C: Transformation matrix between descriptors.

    """
    p_inv_phi_A = torch.pinverse(phi_A)
    p_inv_phi_B = torch.pinverse(phi_B)

    c_G_A = torch.matmul(p_inv_phi_A, G_A)
    c_G_B = torch.matmul(p_inv_phi_B, G_B)

    C = torch.matmul(c_G_A, torch.pinverse(c_G_B))
    P = torch.matmul(
        phi_A,
        torch.matmul(
            C,
            torch.matmul(torch.transpose(phi_B, 2, 1), area),
        ),
    )
    Q = F.normalize(P, 2, 1) ** 2

    return Q, C


# Training Loop for 800 epochs
for epoch in range(800):
    # Step the learning rate scheduler
    scheduler.step()

    # Initialize training loss for the epoch
    train_loss = 0
    basisNet.train()

    # Training phase
    for data in tqdm(train_loader, desc=f"Epoch {epoch} - Training"):
        points, dist, desc, area_tensor = (
            data[0],
            data[1],
            data[2],
            data[4],
        )
        points, dist, area_tensor = (
            points.to(device),
            dist.to(device),
            area_tensor.to(device),
        )
        desc = torch.Tensor(desc).to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass through the model
        pred, _, _ = basisNet(points)

        # Generate pairs for loss calculation
        basis_A, basis_B = pred[:-1], pred[1:]
        dist_x, dist_y = dist[:-1], dist[1:]
        desc_A, desc_B = desc[:-1], desc[1:]
        area_B = area_tensor[1:]

        # Calculate descriptor loss
        Q, C = desc_loss(basis_A, basis_B, desc_A, desc_B, area_B)

        # Calculate individual loss components

        # Distortion Minimisation Term
        eucl_loss_1 = criterion(
            torch.bmm(Q.transpose(2, 1), torch.bmm(dist_x, Q)),
            dist_y,
        )

        # Functional Map Orthogonality Loss with weightage of 0.1
        eucl_loss_2 = 0.1 * criterion(
            torch.bmm(C.transpose(2, 1), C),
            iden_1,
        )

        # Basis Orthogonality Loss with weightage of 0.1
        eucl_loss_3 = 0.1 * criterion(
            torch.bmm(
                pred.transpose(2, 1),
                torch.bmm(area_tensor, pred),
            ),
            iden_1,
        )

        # Total loss
        loss = eucl_loss_1 + eucl_loss_2 + eucl_loss_3

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate training loss
        train_loss += loss.item()

    print(f"Epoch {epoch} - Training Loss: {train_loss}")

    # Validation phase
    eval_loss = 0
    basisNet.eval()

    # Disable gradient calculation for validation
    with torch.no_grad():
        for data in tqdm(
            test_loader,
            desc=f"Epoch {epoch} - Validation",
        ):
            points, dist, desc, area_tensor = (
                data[0],
                data[1],
                data[2],
                data[4],
            )
            points, dist, area_tensor = (
                points.to(device),
                dist.to(device),
                area_tensor.to(device),
            )
            desc = torch.Tensor(desc).to(device)

            # Forward pass through the model
            pred, _, _ = basisNet(points)

            # Generate pairs for loss calculation
            basis_A, basis_B = pred[:-1], pred[1:]
            dist_x, dist_y = dist[:-1], dist[1:]
            desc_A, desc_B = desc[:-1], desc[1:]
            area_B = area_tensor[1:]

            # Calculate descriptor loss
            Q, C = desc_loss(basis_A, basis_B, desc_A, desc_B, area_B)

            # Calculate individual loss components

            # Distortion Minimisation Term
            eucl_loss_1 = criterion(
                torch.bmm(Q.transpose(2, 1), torch.bmm(dist_x, Q)),
                dist_y,
            )

            # Functional Map Orthogonality Loss with weightage of 0.1
            eucl_loss_2 = 0.1 * criterion(
                torch.bmm(C.transpose(2, 1), C),
                iden_1,
            )

            # Basis Orthogonality Loss with weightage of 0.1
            eucl_loss_3 = 0.1 * criterion(
                torch.bmm(
                    pred.transpose(2, 1),
                    torch.bmm(area_tensor, pred),
                ),
                iden_1,
            )

            # Total loss
            loss = eucl_loss_1 + eucl_loss_2 + eucl_loss_3
            eval_loss += loss.item()

    print(f"Epoch {epoch} - Validation Loss: {eval_loss}")

    # Save the model if it achieves the best evaluation loss so far
    if eval_loss < best_eval_loss:
        best_eval_loss = eval_loss
        torch.save(
            basisNet.state_dict(),
            f"{OUT_DIR}/basis_model_epoch_{epoch}.pth",
        )
        print("Model saved.")
