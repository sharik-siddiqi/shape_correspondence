from code.dataloader import pc_normalize

import igl
import numpy as np
import torch
from model import PointNetBasis, PointNetDesc
from our_match import our_match, our_match_desc
from scipy.io import loadmat, savemat

from evaluation.compute_gdsc_error import (
    compute_error_curves,
    compute_geodesic_error,
)

# Set device to CPU
device = torch.device("cpu")

# Load models
basis_model = PointNetBasis(k=20, feature_transform=False).to(device)
desc_model = PointNetDesc(k=40, feature_transform=False).to(device)

# Path to vertices/co-ordinates for FAUST shapes with 2100 vertices
vertices_path = "<path_to_faust_vertices_remeshed_2100.mat>"

# Path to geodesic distance for FAUST shapes with 2100 vertices
geo_dist_path = "<path_to_faust_geo_dist_2100_1.mat>"

# Downsampling match path for 2100 to 1000 vertices in FAUST shapes
match_path = "<path_to_match_faust_2100_1000.mat>"

# Path to mesh faces for 1000 vertices FAUST shapes
mesh_faces_path = "<path_to_tr_reg_031_simplified_1000.ply>"

# Load data
v = loadmat(vertices_path)
geo_dist = loadmat(geo_dist_path)["geo"]
match = loadmat(match_path)["match_2100_1000"][:, 0] - 1
_, f = igl.read_triangle_mesh(mesh_faces_path)

# Preprocess data
order = np.argsort(v["indices"])
geo_dist = geo_dist[order[0, :], :, :]
v_clean = v["vertices_clean"][order[0, :], :, :]
v_clean = v_clean[:, match, :]
geo_dist = geo_dist[:, match, :][:, :, match]

# Normalize vertices
for i in range(v_clean.shape[0]):
    v_clean[i, :, :] = pc_normalize(v_clean[i, :, :])

# Generate source and target indices for isometric cases
src, tar = [], []
for i in range(10):
    for j in range(1, 10):
        s = list(range(i * 10, (i + 1) * 10))
        t = s[j:] + s[:j]
        src.extend(s)
        tar.extend(t)
src, tar = np.array(src), np.array(tar)

# Load model checkpoints
basis_model.load_state_dict(
    torch.load("<path_to_basis_model_checkpoint.pth>"),
)
desc_model.load_state_dict(
    torch.load("<path_to_desc_model_checkpoint.pth>"),
)

# Set models to evaluation mode
basis_model.eval()
desc_model.eval()

# Compute basis and descriptors
v_tensor = (
    torch.from_numpy(v_clean.astype(np.float32))
    .transpose(1, 2)
    .to(device)
)

# Compute basis and descriptors
pred_basis = basis_model(v_tensor)[0].detach().cpu().numpy()
pred_desc = desc_model(v_tensor)[0].detach().cpu().numpy()

# Perform matching and compute geodesic errors
geo_err_main = []
match_desc_all = []  # To store all our_match_desc values
for k in range(len(src)):
    phiM, phiN = pred_basis[src[k]], pred_basis[tar[k]]
    descM, descN = pred_desc[src[k]], pred_desc[tar[k]]

    # Compute correspondence using our_match and our_match_desc
    match_opt = our_match(phiM, phiN)
    match_desc = our_match_desc(phiM, phiN, descM, descN)[0]
    match_desc_all.append(match_desc)  # Store our_match_desc values

    geo_dist_case = geo_dist[tar[k], :, :]

    # Compute geodesic errors
    errors = compute_geodesic_error(
        [match_opt, match_desc],
        geo_dist_case,
    )
    geo_err_main.append(errors[:, 1])

# Compute mean curves and save results
geo_err_main = np.array(geo_err_main)
match_desc_all = np.array(match_desc_all)  # Convert to numpy array
thr = np.linspace(0, 1, 1000)
mean_curves = compute_error_curves(geo_err_main, thr) / len(src)

# Save geodesic error curves and our_match_desc values
savemat(
    "<path_to_save_curve_geo_error.mat>",
    {
        "mean_curves": mean_curves,
        "thr": thr,
        "geo_err_main": geo_err_main,
        "match": match_desc_all,  # Save our_match_desc values
    },
)

# Save descriptor values
savemat(
    "<path_to_save_descriptor_values.mat>",
    {"descriptors": pred_desc},
)

print("Evaluation completed!")
