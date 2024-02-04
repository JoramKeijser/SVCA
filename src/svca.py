"""
Shared Variance Component Analysis (SVCA)
from Stringer et al., Science 2019, https://doi.org/10.1126/science.aav7893
"""
import numpy as np
from scipy.linalg import svd
import warnings

def svca(Ftrain, Ftest, Gtrain, Gtest, n_dims = None):
    """
    Shared Variance Component Analysis (SVCA, Stringer et al., Science 2019)
    to estimate the dimensionality of neural activity using cross-validation
    across observations/time bins and features/neurons.
    By definition, reliable variance generalizes across both obervations and features

    Arguments:
        Ftrain (array): (train samples, neurons set 1)
        Ftest (array):(test samples, neurons set 1)
        Gtrain (array):(train samples, neurons set 2)
        Gtest (array):(test samples, neurons set 2)
        n_dims (int): number of dimensions, default - all dimensions

    Returns:
        reliabe_variance (n_dims, ): reliable variance per dimension
        all_variance (n_dims, ): mean of variance within neuron groups
        SVC1 (time, n_dims): test observations of neuron group 1 projected on to SVCs
        SVC2 (time, n_dims): test observations of neuron group 2 projected on to SVCs

    """
    assert Ftrain.shape[1] == Ftest.shape[1] # neurons set 1
    assert Gtrain.shape[1] == Gtest.shape[1] # neurons set 2
    assert Ftrain.shape[0] == Gtrain.shape[0] # samples
    assert Ftest.shape[0] == Gtest.shape[0] # samples
    if n_dims is None: # default - pick maximum number of dimensions
        n_dims = min(Ftrain.shape[1], Gtrain.shape[1])
    # Covariance of train and test samples
    Ttrain = Ftrain.shape[0]
    Ttest = Ftest.shape[0]
    Ctrain = Ftrain.T @ Gtrain / Ttrain  # N_train x N_test
    Ctest = Ftest.T @ Gtest / Ttest
    # SVD of train covariance to find bases for neuron groups
    U, _, Vh = svd(Ctrain)
    # Test variance within groups
    C_F = Ftest.T @ Ftest / Ttest
    C_G = Gtest.T @ Gtest / Ttest
    # Variance captured by bases vectors
    S_hat = U[:,:n_dims].T @ Ctest @ Vh[:n_dims].T # generally not diagonal
    S_F = U[:,:n_dims].T @ C_F @ U[:,:n_dims] # test var - one set of neurons
    S_G = Vh[:n_dims] @ C_G @ Vh[:n_dims].T # the other
    # Reliable var: what generalizes from train to test
    reliable_variance = np.diag(S_hat)
    # Within group variance
    all_variance = (0.5 * np.diag(S_F + S_G))
    # Project test data onto bases vecs
    SVC1 = Ftest @ U[:,:n_dims] # (time, neurons) * (neurons, pcs)
    SVC2 = Gtest @ Vh[:n_dims].T

    return reliable_variance, all_variance, SVC1, SVC2


def split_data(X, position, bin_width = None, neuron_bins = 16, time_bins = 60, shuffle = False, seed = None):
    """
    Split data along time/observation and neuron/feature axes and subtract mean activity for each neuron

    Arguments:
        X (array): observations (time bins) x features (neurons)
        position (array): position of each neuron - should be x or y direction
            to avoid overlap in z-direction
        bin_width (float): width (same units as position) of bins in
            which to divide neurons. If None - use neuron_bins
        neuron_bins (int): number of bins/strips - ignored unless bin_width is None
        time_bins (int): width (same units as time axis) of time bins
        shuffle (bool): shuffle observations for each neuron
        seed (int): for fixing randomness - ignored unless shuffle is True

    Returns:
        Ftrain (train samples, neurons set 1)
        Ftest (test samples, neurons set 1)
        Gtrain (train samples, neurons set 2)
        Gtest (test samples, neurons set 2)
    """
    time_steps, neurons = X.shape
    if shuffle: # Permute observations of each neuron
        rng = np.random.RandomState(seed)
        n_neurons = X.shape[1]
        for i in range(neurons):
            X[:,i] = np.random.permutation(X[:,i])
    # Split neurons
    if bin_width is None:
        bin_width = (position.max() - position.min()) / neuron_bins
    else:
        warnings.warn(f"Using bin_width variable to split neurons; ignoring neuron_bins")
    if not shuffle and seed is not None:
        warnings.warn(f"Ignoring provided random seed {seed}")
    # Create bins spanning spatial positions
    bins = np.arange(position.min(), position.max()+bin_width, bin_width)
    # Combine neurons into alternating even/odd groups
    neuron_id = np.digitize(position, bins) % 2 == 0 # assign alternating bins to same group
    # Split observations (time) in two alternating bins
    split_id = np.ones((2*time_bins,)) # Index for first 2 alternating bins
    split_id[:time_bins] = 0
    # Extend to >=2 bins to include all observations
    n_reps = np.round(time_steps/(2*time_bins)) +1
    time_id = np.tile(split_id, int(n_reps))[:time_steps]
    # Now we can split along both axes
    Ftrain = X[time_id == 0][:,neuron_id]
    Ftest = X[time_id == 1][:,neuron_id]
    Gtrain = X[time_id == 0][:,~neuron_id]
    Gtest = X[time_id == 1][:,~neuron_id]
    # Center each neuron
    mu_F = Ftrain.mean(0, keepdims=True)
    Ftrain -= mu_F
    Ftest -= mu_F
    mu_G = Gtrain.mean(0, keepdims=True)
    Gtrain -= mu_G
    Gtest -= mu_G

    return Ftrain, Ftest, Gtrain, Gtest
