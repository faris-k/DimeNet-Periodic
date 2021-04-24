# Directional Message Passing Neural Network (DimeNet) Applied to Periodic Structures

This repository is a PyTorch version of [DimeNet++](https://github.com/klicperajo/dimenet). It is based on the [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric) implementation of DimeNet with a few corrections and updates from the TensorFlow repo to reflect the status of DimeNet++.

The focus of this work is to apply DimeNet's directional message-passing to periodic systems (inorganic crystals). Element embeddings are used as the only additional feature to DimeNet++, which normally uses just structural and compositional information to predict properties.

Materials data from the [Matbench benchmark dataset](https://hackingmaterials.lbl.gov/automatminer/datasets.html) is used to assess performance in predicting materials properties.

This work was supported by funding from the Undergraduate Research Opportunities Program at the University of Utah. Please note that this is very much a work in progress.

## Usage
Run `data_parsing.py` first to create parsed datasets from the Matbench benchmark datasets. Then run `train.py`.

## Issues
On some systems, there may be incompatibilities between PyTorch Geometric and [Pymatgen](https://github.com/materialsproject/pymatgen) installed to the same environment. At least, this was the case with my testing system. To get around this, create two separate environments:
* A data parsing environment with Pymatgen installed to run `data_parsing.py`.
* An experimentation environment with PyTorch Geometric installed to run `train.py` and any other files.
