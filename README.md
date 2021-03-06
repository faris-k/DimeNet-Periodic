# Directional Message Passing Neural Network (DimeNet) Applied to Periodic Structures

This repository is a PyTorch version of [DimeNet++](https://github.com/klicperajo/dimenet). It is based on the [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric) implementation of DimeNet with a few corrections and updates from the TensorFlow repo to reflect the status of DimeNet++.

The focus of this work is to apply DimeNet's directional message-passing to periodic systems (inorganic crystals). Materials data from the [Matbench benchmark dataset](https://hackingmaterials.lbl.gov/automatminer/datasets.html) is used to assess performance in predicting materials properties.

This work was supported by funding from the Undergraduate Research Opportunities Program at the University of Utah. Please note that this is very much a work in progress.

## Performance Metrics
Matbench datasets were split using five-fold nested cross-validation as described by [Matbench v0.1 documentation](https://hackingmaterials.lbl.gov/automatminer/datasets.html#benchmarking-and-reporting-your-algorithm). Each fold has a 60-20-20 split for training, validation, and test data respectively.

|Matbench Dataset|Target Property|MAE|
|---|---|---|
|`matbench_jdft2d`|DFT Exfoliation Energy|44.846 meV/atom|
|`matbench_phonons`|Phonon Peak|51.074 1/cm|
|`matbench_dielectric`|Refractive Index|0.344 (unitless)|
|`matbench_log_kvrh`|Bulk Modulus (log10)|0.0666 log(GPa)|
|`matbench_log_gvrh`|Shear Modulus (log10)|0.0900 log(GPa)|
|`matbench_perovskites`|Formation Energy|0.0437 eV/unit cell|


## Usage
Run `data_parsing.py` first to create parsed datasets from the Matbench benchmark datasets. I suggest parsing only the smaller Matbench datasets first. Then run `train.py`. Training may be a little slow, since DimeNet's message passing scheme is computationally expensive. I suggest keeping batch size small (16 or less) to avoid CUDA memory issues. For reference, it takes 14.54 GB of VRAM for a batch size of 16 on the `matbench_mp_gap` dataset (about 63,680 structures in the training set).

A Google Colab notebook is provided in `New_Dimenet.ipynb` that shows an entire training run. In the notebook, datasets are loaded onto Google Drive, so running the file as-is won't work without first correcting the file directories to suit your needs.

### Possible Compatibility Issues
On some systems, there may be incompatibilities between PyTorch Geometric and [Pymatgen](https://github.com/materialsproject/pymatgen) installed to the same environment. At least, this was the case with my testing system. To get around this, I suggest create two separate environments:
* A data parsing environment with Pymatgen installed to run `data_parsing.py` (or `data_parsing.ipynb`). In a new environment, install [pymatgen](https://pymatgen.org/installation.html#step-3-install-pymatgen) and [matminer](https://hackingmaterials.lbl.gov/matminer/installation.html). See `parse_requirements.txt` and `cif-parse.yml` for all dependencies.
* An experimentation environment with PyTorch Geometric installed to run `train.py` (or `New_Dimenet.ipynb`). This will require [PyTorch](https://pytorch.org/get-started/locally/), then [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric#installation). See `train_requirements.txt` and `dimenet4.yml` for all dependencies.
