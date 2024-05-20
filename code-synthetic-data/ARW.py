#ARW functions
import numpy as np

#compute cdfhat(q)
def cdfhat(q, arr):
    count_leq = np.count_nonzero(arr <= q) 
    quantile = count_leq / len(arr)  
    return quantile

def cdfhat_arr(q, arr):
    q = q[:, np.newaxis]  # Convert q to column vector for broadcasting
    cdf_values = np.mean(arr <= q, axis=1)  # Broadcast comparison over arr and calculate mean
    return cdf_values

#compute psi_tk for all k
def compute_psi(Btk_all, alpha, delta):
    sigma_bin = alpha *(1-alpha) 

    #NOTE: this part can be changed
    psi_all = np.sqrt(2 * sigma_bin * np.log(1/delta) / Btk_all)
    #psi_1_all = 1.5 * np.sqrt(2 * sigma_bin * np.log(4/delta) / Btk_all) + (19/4) * np.log(4/delta) / Btk_all
    #psi_2_all = np.sqrt(2 * sigma_bin * np.log(4/delta) / Btk_all) + (19/6) * np.log(4/delta) / Btk_all

    return psi_all

# Compute summary statistics

def prepare(U, B_arr, alpha):

    """Compute B_tk, mu_hat_tk and v_hat_tk for all k
    Args:
        U (np.array): 1D array of length B_1 + ... + B_t
        B_arr: (np.array) stores the sample size for each period
        alpha: quantile level
        delta: exceptional probability
    Returns:
        Btk_all, qtk_all, Fti_qtk_all
    """

    period_ends = np.cumsum(B_arr)
    period_starts = period_ends - B_arr

    t = len(B_arr)
    
    B_rev = B_arr[::-1]
    Btk_all = np.cumsum(B_rev)

    #compute empirical quantiles for each k 
    qtk_all = np.array( [np.quantile( U[period_starts[i]:], 1-alpha, method='inverted_cdf') for i in range(t)] )[::-1]

    Fti_qtk_all = np.zeros((t,t))

    for i in range(t):
        Fti_qtk_all[i,i:] = cdfhat_arr(qtk_all[i:], U[period_starts[t-i-1]:])

    return Btk_all, qtk_all, Fti_qtk_all

# Goldenshluger-Lepski procedure

def ARWQE(U, B_arr, alpha, delta, gamma=1):

    """ Adaptive Rolling Window Mean Estimation(ARWME): selecting the best window size
    Args:
        U (np.array): 1D array of length B_1 + ... + B_t
        B_arr: (np.array) stores the sample size for each period
        alpha: quantile level
        delta: exceptional probability

    Returns:
        k_hat +1 (int): best window size
        qtk_all[k_hat] (float): best quantile estimate
    """

    # Compute B_tk, mu_tk and v_tk for all k
    Btk_all, qtk_all, Fti_qtk_all = prepare(U, B_arr, alpha)

    t = len(B_arr)
    
    psi_all = compute_psi(Btk_all, alpha, delta)
    # Compute phi_hat_k for all k
    phi_hat_all = np.empty(t)
    for k in range(t):

        #NOTE: this part can be changed
        tmp = np.array([ np.abs(Fti_qtk_all[i,k] - (1-alpha)) - (psi_all[k] - psi_all[i])*gamma for i in range(k + 1) ])
        #print(tmp<0)
        tmp[tmp < 0] = 0
        phi_hat_all[k] = np.max(tmp)

    # Choose k_hat
    k_hat = np.argmin(phi_hat_all + psi_all)
    return k_hat+1, qtk_all[k_hat], qtk_all