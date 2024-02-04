# SVCA
Shared Variance Component Analysis (SVCA, [Stringer et al.](https://doi.org/10.1126/science.aav7893)) is a
clever but simple method to estimate the dimensionality of (neural) data. This repository provides a
Python-based implementation, along with an application to data from the original paper,
and to synthetic data as a sanity check.

Briefly, SVCA estimates reliable variance using cross-validation along both the observation (time) and feature (neuron) axes.
First, the covariance between two (spatially) non-overlapping groups of neurons is computed, based on their activity during training time points. A singular value decomposition (SVD) of the covariance is used to compute orthogonal bases for both sets of activity. The amount of shared or reliable variance is then defined as the variance captured by this basis in _test_ samples (time points). This variance can be compared to the variance each of the bases captures within the two neural populations. Key metrics for a dataset's dimensionality are therefore the number of components required to capture at least, say, 85% of the total reliable variance, and the fraction of reliable variance among those components.


Refer to the methods section of the original paper for more details:
> Stringer, C., Pachitariu, M., Steinmetz, N., Reddy, C. B., Carandini, M., & Harris, K. D. (2019). Spontaneous behaviors drive
> multidimensional, brainwide activity. Science, 364(6437), eaav7893.


## Installation
Create mamba (or conda) environment with necessary Python packages:
```
mamba env create --name svca --file environment.yml
```
Install project package (e.g., to access ``src/`` code):
```
pip install -e .
```

## Download data
From the command line:
```
mkdir data/
wget -O data/stringer_spontaneous.npy https://osf.io/dpqaj/download
```

## Acknowledgements
Thanks to Carsen Stringer and colleagues for sharing their data, see [Figshare](https://doi.org/10.25378/janelia.6163622.v6).
