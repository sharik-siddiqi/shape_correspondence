# Guidance for Configuring Paths in `train_basis.py` and `train_desc.py`

Before running the `train_basis.py` script, ensure the following paths are correctly configured in the script:

1. **Path to a sample mesh in the training dataset**:
   - Update the `simple_mesh_path` variable with the path to a sample mesh file from the training dataset.
   - Example:
     ```python
     simple_mesh_path = "path/to/your/sample_mesh.mat"
     ```

2. **Path to the training dataset**:
   - Update the `DATA_PATH` variable with the root directory of your training dataset.
   - Example:
     ```python
     DATA_PATH = "path/to/your/train_dataset"
     ```

3. **Output directory for saving trained models**:
   - The script saves trained models in the `OUT_DIR` directory. Ensure this directory exists or is created during runtime.
   - Example:
     ```python
     OUT_DIR = "./models/trained"
     ```

4. **Descriptor files (SHOT, HKS, WKS)**:
   - For our experimentation, we're using SHOT, Heat Kernel Signature (HKS), or Wave Kernel Signature (WKS) descriptors, ensure the paths to these files are correctly set in the script.
   - Example:
     ```python
     shot_0 = loadmat('./path/to/shot_surreal.mat')  # SHOT descriptors
     hks_0 = loadmat('./path/to/hks_surreal.mat')    # HKS descriptors
     wks_0 = loadmat('./path/to/wks_surreal.mat')    # WKS descriptors
     ```

To train the basis model with any specific dataset (after making necessary changes), run these commands:

```train basis network
python .\code\train_basis.py
```

Before running the `train_desc.py` script, ensure the following paths are correctly configured in the script:

1. **Path to the training dataset**:
   - Update the `DATA_PATH` variable with the root directory of your training dataset.
   - Example:
     ```python
     DATA_PATH = "path/to/your/train_dataset"
     ```

2. **Output directory for saving trained models**:
   - The script saves trained models in the `OUT_DIR` directory. Ensure this directory exists or is created during runtime.
   - Example:
     ```python
     OUT_DIR = "./models/trained"
     ```

3. **Pretrained weights for the `PointNetBasis` model**:
   - Ensure the path to the pretrained `basis_model.pth` file is correctly set in the script.
   - Example:
     ```python
     checkpoint = torch.load(f"{OUT_DIR}/basis_model.pth")
     ```
To train the basis model with any specific dataset (after making necessary changes), run these commands:

```train descriptor network
python .\code\train_desc.py
```
