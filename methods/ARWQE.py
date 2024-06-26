#ARW functions
import numpy as np
from scipy.stats import norm

def create_window_sizes(n):
    k_sizes =  [2**i for i in range(int(np.log2(n))+1)]
    if k_sizes[-1] < n:
        k_sizes.append(n)
    return np.array(k_sizes).astype(int)

def cdfhat_arr(q, arr):
    q = q[:, np.newaxis]  # Convert q to column vector for broadcasting
    cdf_values = np.mean(arr <= q, axis=1)  # Broadcast comparison over arr and calculate mean
    return cdf_values

#compute psi_tk for all k
def compute_psi(Btk_all, alpha, delta):
    v = alpha * (1 - alpha)
    psi_all = np.sqrt(v * np.log(1/delta) / Btk_all) + 1 / Btk_all
    return psi_all

# Compute summary statistics
def prepare(U, B_arr, alpha):

    period_ends = np.cumsum(B_arr) 
    period_starts = period_ends - B_arr
    t = len(B_arr)

    #create an array of power of 2 smaller than len(B_arr)
    k_sizes = create_window_sizes(t)
    w = len(k_sizes)

    B_rev = B_arr[::-1]
    B_all = np.cumsum(B_rev)
    Btk_all = B_all[k_sizes-1]

    #compute empirical quantiles for each k 
    qtk_all = np.array( [np.quantile( U[period_starts[t-k]:], 1-alpha, method='inverted_cdf') for k in k_sizes] )

    Fti_qtk_all = np.zeros((w, w))

    for idx, i in enumerate(k_sizes):
        Fti_qtk_all[idx,idx:] = cdfhat_arr(qtk_all[idx:], U[period_starts[t-i]:])

    return k_sizes, Btk_all, qtk_all, Fti_qtk_all

# Goldenshluger-Lepski procedure

def ARWQE(U, B_arr, alpha, delta, gamma=1):

    # Compute B_tk, mu_tk and v_tk for all k
    k_sizes, Btk_all, qtk_all, Fti_qtk_all = prepare(U, B_arr, alpha)

    w = len(k_sizes)
    
    psi_all = compute_psi(Btk_all, alpha, delta)

    # Compute phi_hat_k for all k
    phi_hat_all = np.empty(w)

    for k in range(w):

        tmp = np.array([ np.abs(Fti_qtk_all[i,k] - (1-alpha)) - (psi_all[k] + psi_all[i])*gamma for i in range(k + 1) ])
        tmp[tmp < 0] = 0
        phi_hat_all[k] = np.max(tmp) * (5 / 12)

    # Choose k_hat
    k_hat = np.argmin(phi_hat_all + psi_all)
    return k_sizes[k_hat], qtk_all[k_hat], qtk_all

def calculate_coverage(y_hat, qt_khat, mu_t, variance):
    y_upp = y_hat + qt_khat
    y_low = y_hat - qt_khat
    coverage = norm.cdf(y_upp, mu_t, np.sqrt(variance)) - norm.cdf(y_low, mu_t, np.sqrt(variance))
    return coverage


#######################

def QE_fixed(U, B_arr, alpha, window):
    T = len(B_arr)
    k = min(window, T)
    start_k = np.sum(B_arr[0:(T - k)])
    q_hat = np.quantile( U[start_k:], 1 - alpha, method='inverted_cdf')
    return q_hat


def weighted_quantile(x, q, weights): # compute weighted quantile
    sorter = np.argsort(x)
    sorted_values = x[sorter]
    sorted_weights = weights[sorter]
    cumulative_weights = np.cumsum(sorted_weights)
    percentile_cutoffs = np.quantile(cumulative_weights, q, method = 'inverted_cdf')
    return np.interp(percentile_cutoffs, cumulative_weights, sorted_values)


def QE_weighted(U, B_arr, alpha, rho):
    period_ends = np.cumsum(B_arr) 
    period_starts = period_ends - B_arr
    T = len(B_arr)
    weights = np.ones(len(U))
    for t in range(1, T):
        weights[period_starts[T - 1 - t]:period_ends[T - 1 - t]] = (rho ** t)
    q_hat = weighted_quantile( U, 1 - alpha, weights = weights)
    return q_hat


