import numpy as np

#generate synthetic data of given B_arr and true means with Gaussian noise

def generate_gaussian_sequence(B_arr, true_means, variance):

    """generate 1D synthetic data with given B_arr and true means with Gaussian noise
    Args:
        B_arr (np.array): stores the sample size for each period
        true_means (np.array): true means for each period
        variance (float): variance of Gaussian noise for all periods
    Returns:
        X_1D (np.array): 1D array of length B_1 + ... + B_t
    """

    # Generate random data points for each period
    X_list = [np.random.normal(loc=mu, scale=np.sqrt(variance), size=B_i) for mu, B_i in zip(true_means, B_arr)]

    # Combine data into a single 1D array
    X_1D = np.concatenate(X_list)

    return X_1D


#generate true means 

def generate_true_means(N, n, seed=2024):

    """generate true means for synthetic data
    Args:
        N (int): number of periods - 1
        n (int): a unit for phase length
        seed (int): random seed
    Returns:
        theta (np.array): true means for each period
    """
    
    theta = np.zeros(1 + N)
    np.random.seed(seed)

    # phase 1: big shifts
    eta = 0.005
    le = 1
    tmp = le
    for i in range(2 * n):
        theta[le + i] = theta[le + i - 1] + eta
        tmp += 1
    le = tmp
    for i in range(int(n / 2)):
        theta[le + i] = theta[le + i - 1] - eta
        tmp += 1
    le = tmp
    for i in range(int(n / 2)):
        theta[le + i] = theta[le + i - 1]
        tmp += 1
    le = tmp

    # phase 2: sinusoid
    for i in range(2 * n):
        theta[le + i] = theta[le - 1] - 0.1 * np.sin( np.pi * i / (1 * n) )
        tmp += 1
    le = tmp
    for i in range(2 * n):
        theta[le + i] = theta[le - 1] - 0.1 * np.sin( np.pi * i / (3 * n) )
        tmp += 1
    le = tmp

    # phase 3: constant
    for i in range(8 * n):
        theta[le + i] = theta[le - 1] - 0.3
        tmp += 1
    le = tmp

    # phase 4: steps
    m = N - le + 1
    steps = np.random.binomial(1, 0.5, size = m) * 2 - 1
    for i in range(m):
        theta[le + i] = theta[le + i - 1] + 0.02 * steps[i]
        tmp += 1
    le = tmp

    return theta


#for linear regression

def generate_multinomial_X(meanX, dimX, N):
    return np.random.multivariate_normal(meanX*np.ones(dimX), np.eye(dimX), N)

def generate_linreg_data(meanX, B_arr, beta, variance):

    dimX = beta.shape[1]
    num_periods = len(B_arr)
    B_arr_starts = np.cumsum(B_arr) - B_arr
    B_arr_ends = np.cumsum(B_arr) - 1

    X = generate_multinomial_X(meanX, dimX, sum(B_arr))

    mu_y = np.zeros(len(X))

    for t in range(num_periods):
        X_t = X[B_arr_starts[t]:B_arr_ends[t]+1]
        mu_y[B_arr_starts[t]:B_arr_ends[t]+1] = X_t @ beta[t].T

    y = mu_y + np.random.normal(0, np.sqrt(variance), len(mu_y))

    return X, y

from sklearn.linear_model import LinearRegression

def fit_and_get_scores(X_tr, y_tr, X_val, y_val):
    reg = LinearRegression().fit(X_tr, y_tr)
    S = np.abs(y_val - reg.predict(X_val))
    return reg, S

def monte_carlo_coverage(y_hat, q, Y_test):
    y_upp = y_hat + q
    y_low = y_hat - q
    coverage = np.mean((Y_test <= y_upp) & (Y_test >= y_low))
    return coverage

def check_coverage(y_hat, q, y_test):
    y_upp = y_hat + q
    y_low = y_hat - q
    coverage = (y_test <= y_upp) & (y_test >= y_low)
    return coverage

def fit_LS_get_scores(X_tr, Y_tr, X_val, Y_val):
    beta_hat = np.linalg.solve(X_tr.T.dot(X_tr), X_tr.T.dot(Y_tr))
    S = np.abs(X_val.dot(beta_hat) - Y_val)
    return beta_hat, S