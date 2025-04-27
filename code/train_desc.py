# Import necessary libraries
import os
import random
from code.dataloader import (
    Surr12kModelNetDataLoader as DataLoader,
)

import numpy as np
import torch
import torch.nn.functional as F
from model import PointNetBasis, PointNetDesc
from torch import nn, optim
from tqdm import tqdm

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seed for reproducibility
manualSeed = 1
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Batch size for training and validation
b_size = 10

# Create output directory for saving trained models
OUT_DIR = "./models/trained"
os.makedirs(OUT_DIR, exist_ok=True)

# Path to dataset
DATA_PATH = "path/to/your/train_dataset"

# Load training and testing datasets
TRAIN_DATASET = DataLoader(
    root=DATA_PATH,
    npoint=1000,
    split="train",
    uniform=True,
    normal_channel=False,
    augm=True,
)
TEST_DATASET = DataLoader(
    root=DATA_PATH,
    npoint=1000,
    split="test",
    uniform=True,
    normal_channel=False,
    augm=True,
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

# Initialize the PointNetBasis model and load pretrained weights
basis = PointNetBasis(k=20, feature_transform=False)
checkpoint = torch.load(f"{OUT_DIR}/basis_model.pth")
basis.load_state_dict(checkpoint)
basis.to(device)

# Initialize the PointNetDesc model
descriptor = PointNetDesc(k=40, feature_transform=False)
descriptor.to(device)

# Define optimizer and learning rate scheduler
optimizer = optim.Adam(
    descriptor.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
)
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=100,
    gamma=0.8,
)

# Define loss function
criterion = nn.MSELoss(reduction="mean")

# Generate identity matrix for loss calculation
iden_1 = np.identity(20).reshape(1, 20, 20)
iden_1 = torch.Tensor(np.repeat(iden_1, b_size, axis=0)).to(device)

# Track best evaluation loss
best_eval_loss = np.inf


# Descriptor loss function
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


# Training loop
for epoch in range(800):
    # Step the learning rate scheduler
    scheduler.step()

    # Initialize training loss for the epoch
    train_loss = 0
    descriptor.train()

    # Training phase
    for data in tqdm(train_loader, desc=f"Epoch {epoch} - Training"):
        points, dist, area_tensor = data[0], data[1], data[4]
        points, dist, area_tensor = (
            points.to(device),
            dist.to(device),
            area_tensor.to(device),
        )

        # Transpose points for compatibility with the model
        points = points.transpose(2, 1)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass through the basis model (frozen)
        with torch.no_grad():
            basis.eval()
            pred, _, _ = basis(points)

        # Generate pairs for loss calculation
        basis_A, basis_B = pred[:-1], pred[1:]
        dist_x, dist_y = dist[:-1], dist[1:]
        area_B = area_tensor[1:]

        # Forward pass through the descriptor model
        desc, _, _ = descriptor(points)
        desc_A, desc_B = desc[:-1], desc[1:]

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

        # Total loss
        loss = eucl_loss_1 + eucl_loss_2

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate training loss
        train_loss += loss.item()

    print(f"Epoch {epoch} - Training Loss: {train_loss}")

    # Validation phase
    eval_loss = 0
    descriptor.eval()
    with torch.no_grad():
        for data in tqdm(
            test_loader,
            desc=f"Epoch {epoch} - Validation",
        ):
            points, dist, area_tensor = data[0], data[1], data[4]
            points, dist, area_tensor = (
                points.to(device),
                dist.to(device),
                area_tensor.to(device),
            )

            # Transpose points for compatibility with the model
            points = points.transpose(2, 1)

            # Forward pass through the basis and descriptor models
            basis.eval()
            pred, _, _ = basis(points)
            basis_A, basis_B = pred[:-1], pred[1:]
            desc, _, _ = descriptor(points)
            desc_A, desc_B = desc[:-1], desc[1:]
            dist_x, dist_y = dist[:-1], dist[1:]
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

            # Total loss
            loss = eucl_loss_1 + eucl_loss_2
            eval_loss += loss.item()

    print(f"Epoch {epoch} - Validation Loss: {eval_loss}")

    # Save the model if it achieves the best evaluation loss so far
    if eval_loss < best_eval_loss:
        best_eval_loss = eval_loss
        torch.save(
            descriptor.state_dict(),
            f"{OUT_DIR}/desc_model.pth",
        )
        print("Model saved.")
