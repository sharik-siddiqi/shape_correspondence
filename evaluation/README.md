 # Guidance for Configuring Paths in `evaluation_faust.py`

The `evaluation_faust.py` script requires several paths to be configured for loading data, models, and saving results. Below is a detailed explanation of each path and how to set it up.

---

## Paths to Configure

### 1. **Path to FAUST Vertices**
   - **Variable**: `vertices_path`
   - **Description**: This path points to the `.mat` file containing the remeshed FAUST vertices with 2100 vertices.
   - **Example**:
     ```python
     vertices_path = "/path/to/faust_vertices_remeshed_2100.mat"
     ```

---

### 2. **Path to Geodesic Distance Matrix**
   - **Variable**: `geo_dist_path`
   - **Description**: This path points to the `.mat` file containing the geodesic distance matrix for FAUST shapes with 2100 vertices.
   - **Example**:
     ```python
     geo_dist_path = "/path/to/faust_geo_dist_2100_1.mat"
     ```

---

### 3. **Path to Downsampling Matrix**
   - **Variable**: `match_path`
   - **Description**: This path points to the `.mat` file containing the downsampling match data for mapping 2100 vertices to 1000 vertices in FAUST shapes.
   - **Example**:
     ```python
     match_path = "/path/to/match_faust_2100_1000.mat"
     ```

---

### 4. **Path to Mesh Faces**
   - **Variable**: `mesh_faces_path`
   - **Description**: This path points to the `.ply` file containing the mesh faces for FAUST shapes with 1000 vertices.
   - **Example**:
     ```python
     mesh_faces_path = "/path/to/simplified_1000_vertices_shape.ply"
     ```

---

### 5. **Path to Basis Model Checkpoint**
   - **Variable**: `<path_to_basis_model_checkpoint.pth>`
   - **Description**: This path points to the `.pth` file containing the pretrained weights for the `PointNetBasis` model.
   - **Example**:
     ```python
     basis_model.load_state_dict(torch.load("/path/to/basis_model_checkpoint.pth"))
     ```

---

### 6. **Path to Descriptor Model Checkpoint**
   - **Variable**: `<path_to_desc_model_checkpoint.pth>`
   - **Description**: This path points to the `.pth` file containing the pretrained weights for the `PointNetDesc` model.
   - **Example**:
     ```python
     desc_model.load_state_dict(torch.load("/path/to/desc_model_checkpoint.pth"))
     ```

---

### 7. **Path to Save Results**
   - **Variable**: `<path_to_save_curve_geo_error.mat>`
   - **Description**: This path specifies where the computed geodesic error curves will be saved as a `.mat` file.
   - **Example**:
     ```python
     savemat("/path/to/save_curve_geo_error.mat", {"mean_curves": mean_curves, "thr": thr})
     ```

### 8. **Path to Save Descriptor Values**
   - **Variable**: `<path_to_save_descriptor_values.mat>`
   - **Description**: This path specifies where the computed descriptor values will be saved as a `.mat` file.
   - **Example**:
     ```python
     savemat("/path/to/save_descriptor_values.mat", {"descriptors": pred_desc})
     ```


---

## Example Configuration

Here is an example of how the paths might look after configuration:

```python
vertices_path = "/data/faust/faust_vertices_remeshed_2100.mat"
geo_dist_path = "/data/faust/faust_geo_dist_2100_1.mat"
match_path = "/data/faust/match_faust_2100_1000.mat"
mesh_faces_path = "/data/faust/tr_reg_031_simplified_1000.ply"
basis_model.load_state_dict(torch.load("/models/basis_model_checkpoint.pth"))
desc_model.load_state_dict(torch.load("/models/desc_model_checkpoint.pth"))
savemat("/results/curve_geo_error.mat", {"mean_curves": mean_curves, "thr": thr})
savemat("/results/descriptor_values.mat", {"descriptors": pred_desc})