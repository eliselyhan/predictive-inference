import numpy as np

# Training - fixed window sizes
def train_synthetic(X_train, B_arr_train, window_sizes_train):

    """ given 1D array of length B_1 + ... + B_T, and a list of m window sizes,
    compute the corresponding running averages

    Args:
        X_train (np.array): 1D array of length B_1 + ... + B_t
        B_arr_train (array): stores the sample size for each period
        window_sizes_train (list): window sizes

    Returns:
        models (list of np.arrays): len = m; each element is a 1D array
    """

    start_idcs = np.cumsum(B_arr_train) - B_arr_train
    models = [] 
    for k in window_sizes_train:
        start = start_idcs[max(len(B_arr_train) - k, 0)]
        models.append( np.mean( X_train[start:] ) )
        
    return models


# Selection - fixed window sizes
def select_synthetic_fixed(X_val, B_arr_val, models, window_sizes_val):

    """ evaluate models on the validation data within given windows
    Args:
        X_val (np.array): 1D array of length B_1 + ... + B_t
        B_arr_val (array): stores the sample size for each period
        models (list): list of predictions by different models 
        window_sizes_val (list): list of window sizes for evaluation
    Returns:
        indices_selected (list): indices of selected models
        models_selected (list): selected models
    """

    start_idcs = np.cumsum(B_arr_val) - B_arr_val
    models_selected = []
    indices_selected = []
    for window_size in window_sizes_val:
        start = start_idcs[max(len(B_arr_val) - window_size, 0)]
        losses = np.array([ np.mean( (X_val[start:] - model) ** 2 ) for model in models ])
        r_hat = np.argmin(losses)
        indices_selected.append(r_hat)
        models_selected.append(models[r_hat])

    return indices_selected, models_selected


# selection - Adaptive Rolling Window (ARW)

def select_synthetic_ARW(X_val, B_arr_val, models, delta = 0.1, M = 10, seed = 2024):
    losses = [ (X_val - model) ** 2 for model in models ]
    idx_selected = tournament_selection(losses, B_arr_val, delta, M, seed)
    return idx_selected, models[idx_selected]