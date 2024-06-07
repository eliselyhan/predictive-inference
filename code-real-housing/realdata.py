import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import collections
import matplotlib.pyplot as plt
from sklearn import metrics
from tqdm import tqdm

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from ARWQE import ARWQE, QE_fixed, QE_weighted

# Data:
# https://www.dubaipulse.gov.ae/data/dld-transactions/dld_transactions-open


# preprocessing
def preprocess(df, alpha = 0.05, encoding = 'label'):
    # encoding: 'label' for label encoding (1, 2, 3, ...), 'onehot' for one-hot encoding

    # select transactions of flats
    set_rooms = set(['Studio', '1 B/R', '2 B/R', '3 B/R', '4 B/R'])
    df = df[ (df['procedure_name_en'] == 'Sell') &\
            (df['property_sub_type_en'] == 'Flat')&\
            df['rooms_en'].isin(set_rooms) ]


    # select data between given years
    year_start, year_end = 2008, 2023
    tmp_years = np.array([int(x[6:10]) for x in df['instance_date']])
    df = df.iloc[(tmp_years >= year_start) & (tmp_years <= year_end)]
    df['instance_date'] = pd.to_datetime(df['instance_date'], format = '%d-%m-%Y')
    df = df.sort_values('instance_date')



    # select columns and drop missing values
    list_columns = ['instance_date', 'area_name_en', 'rooms_en',\
                    'has_parking', 'procedure_area', 'actual_worth']
    df = df[list_columns].dropna()


    # number of bedrooms
    dict_rooms = dict()
    for (i, r) in enumerate(set_rooms):
        dict_rooms[r] = i
    df.replace({'rooms_en': dict_rooms}, inplace = True)

    # log transform of housing price
    df['actual_worth'] = np.log(df['actual_worth'])

    # remove outliers
    tmp1_low = np.quantile(df['actual_worth'], alpha / 2)
    tmp1_high = np.quantile(df['actual_worth'], 1 - alpha / 2)
    tmp2_low = np.quantile(df['procedure_area'], alpha / 2)
    tmp2_high = np.quantile(df['procedure_area'], 1 - alpha / 2)

    tmp = np.mean((df['actual_worth'] >= tmp1_low) & (df['actual_worth'] <= tmp1_high) &\
            (df['procedure_area'] >= tmp2_low) & (df['procedure_area'] <= tmp2_high))
    #print('Fraction of remaining data:', tmp)


    df = df[ (df['actual_worth'] >= tmp1_low) & (df['actual_worth'] <= tmp1_high) &\
            (df['procedure_area'] >= tmp2_low) & (df['procedure_area'] <= tmp2_high)  ]

    # encoding
    if encoding == 'label':
        df['area_name_en'] = LabelEncoder().fit_transform(df['area_name_en'])
    if encoding == 'onehot':
        df = pd.get_dummies(df, columns=['area_name_en', ])
        df.replace({False: 0, True: 1}, inplace=True)

    return df



# data splitting
def splitting(df0, prop, seed = 2024):
    df = df0.copy()
    tmp = df['instance_date'].dt.isocalendar()
    df['instance_date'] = tmp['year'] * 100 + tmp['week']
    weeks_counter = collections.Counter( df['instance_date'] )

    B_arr = []
    B_arr_train = []
    B_arr_val = []
    B_arr_test = []
    for key in weeks_counter:
        n = weeks_counter[key]
        n_train = int(n * prop[0])
        n_val = int(n * prop[1])   

        B_arr.append(n)
        B_arr_train.append(n_train)
        B_arr_val.append(n_val)
        B_arr_test.append(n - n_train - n_val)
        
    B_arr = np.array(B_arr)
    B_arr_train = np.array(B_arr_train)
    B_arr_val = np.array(B_arr_val)
    B_arr_test = np.array(B_arr_test)

    idx_train, idx_val, idx_test = sample_idx(B_arr, B_arr_train, B_arr_val, B_arr_test, seed)

    df_train = df.iloc[idx_train]
    df_val = df.iloc[idx_val]
    df_test = df.iloc[idx_test]

    X_train = df_train.drop(['instance_date', 'actual_worth'], axis=1)
    X_val = df_val.drop(['instance_date', 'actual_worth'], axis=1)
    X_test = df_test.drop(['instance_date', 'actual_worth'], axis=1)

    y_train = np.array(df_train['actual_worth'])
    y_val = np.array(df_val['actual_worth'])
    y_test = np.array(df_test['actual_worth'])

    data_train = [X_train, y_train, B_arr_train]
    data_val = [X_val, y_val, B_arr_val]
    data_test = [X_test, y_test, B_arr_test]

    return [data_train, data_val, data_test]



# auxiliary function for data splitting
def sample_idx(B_arr, B_arr_train, B_arr_val, B_arr_test, seed):
    #get the end indices for each period
    end_idcs = np.cumsum(B_arr)
    end_idcs_train = np.cumsum(B_arr_train)
    end_idcs_val = np.cumsum(B_arr_val)
    end_idcs_test = np.cumsum(B_arr_test)

    
    idx_train = np.zeros(sum(B_arr_train))
    idx_val = np.zeros(sum(B_arr_val))    
    idx_test = np.zeros(sum(B_arr_test))

    np.random.seed(seed)
    #in each period, split data into train, validation and test
    for t in range(len(B_arr)):
        end = end_idcs[t] 
        start = end - B_arr[t]

        #randomly split indices into train, assess and test
        idx_t = np.arange(start, end)
        
        #partition the indices into train, assess and test
        idx_train_t = np.random.choice(idx_t, B_arr_train[t], replace=False)
        idx_val_t = np.random.choice(np.setdiff1d(idx_t, idx_train_t), B_arr_val[t], replace=False)
        idx_test_t = np.setdiff1d(idx_t, np.concatenate((idx_train_t, idx_val_t)))

        #get the corresponding data
        idx_train[(end_idcs_train[t] - B_arr_train[t]) : end_idcs_train[t]] = idx_train_t
        idx_val[(end_idcs_val[t] - B_arr_val[t]) : end_idcs_val[t]] = idx_val_t
        idx_test[(end_idcs_test[t] - B_arr_test[t]) : end_idcs_test[t]] = idx_test_t

    return idx_train, idx_val, idx_test


# XGBoost or random forest
def test(data, algo, t, window_sizes_train, window_sizes_infer, weights, test_next = False, alpha = 0.1, delta = 0.1, seed = 2024):
    assert algo == 'XGB' or algo == 'RF'
    assert t >= 0 and t < len(data[0][2])
    if test_next:
        assert t < len(data[0][2]) - 1


    [data_train, data_val, data_test] = data
    [X_train, y_train, B_arr_train] = data_train
    [X_val, y_val, B_arr_val] = data_val
    [X_test, y_test, B_arr_test] = data_test

    scaler = StandardScaler()

    if algo == 'XGB':
        regressor = XGBRegressor(random_state = seed)
    elif algo == 'RF':
        regressor = RandomForestRegressor(random_state = seed)

    end_train = np.sum(B_arr_train[0:(t+1)])
    end_val = np.sum(B_arr_val[0:(t+1)])
    if test_next:
        end_test = np.sum(B_arr_test[0:(t+2)])
    else:
        end_test = np.sum(B_arr_test[0:(t+1)])

    num_train = len(window_sizes_train)
    num_infer = len(window_sizes_infer)
    num_weights = len(weights)
    freq_ARW = np.zeros(num_train)
    width_ARW = np.zeros(num_train)
    freq_fixed = np.zeros((num_train, num_infer))
    width_fixed = np.zeros((num_train, num_infer))
    freq_weighted = np.zeros((num_train, num_weights))
    width_weighted = np.zeros((num_train, num_weights))

    if test_next:
        start_test = end_test - B_arr_test[t + 1]
    else:
        start_test = end_test - B_arr_test[t]
    y_true_test = y_test[start_test:end_test]
    scores_test_baseline = np.abs(y_true_test - np.mean(y_true_test))
    q_hat_baseline = np.quantile( scores_test_baseline, 1 - alpha, method='inverted_cdf')
    width_baseline = 2 * q_hat_baseline

    for i in range(num_train):
        # training
        r = window_sizes_train[i]
        start_train = np.sum(B_arr_train[0:max(t - r + 1, 0)])
        X_train_i = scaler.fit_transform(X_train[start_train:end_train])
        regressor.fit(X_train_i, y_train[start_train:end_train])

        # compute comformity scores of validation data
        X_val_i = scaler.transform(X_val[0:end_val])
        y_pred_val = regressor.predict(X_val_i)
        scores_val = np.abs(y_val[0:end_val] - y_pred_val)

        # compute comformity scores of test data
        X_test_i = scaler.transform(X_test[start_test:end_test])
        y_pred_test = regressor.predict(X_test_i)
        scores_test = np.abs(y_true_test - y_pred_test)
        
        # quantile estimation
        # ARW
        _, q_hat, _ = ARWQE(scores_val, B_arr_val[0:(t+1)], alpha, delta)
        freq_ARW[i] = np.mean(scores_test <= q_hat)
        width_ARW[i] = 2 * q_hat

        # fixed windows
        for (j, k) in enumerate(window_sizes_infer):
            q_hat_k = QE_fixed(scores_val, B_arr_val[0:(t+1)], alpha, k)
            freq_fixed[i, j] = np.mean(scores_test <= q_hat_k)
            width_fixed[i, j] = 2 * q_hat_k

        # weighted
        for (j, rho) in enumerate(weights):
            q_hat_rho = QE_weighted(scores_val, B_arr_val[0:(t+1)], alpha, rho)
            freq_weighted[i, j] = np.mean(scores_test <= q_hat_rho)
            width_weighted[i, j] = 2 * q_hat_rho

    
    results = dict()
    results['freq_ARW'] = freq_ARW
    results['freq_fixed'] = freq_fixed
    results['freq_weighted'] = freq_weighted
    results['width_ARW'] = width_ARW
    results['width_fixed'] = width_fixed
    results['width_weighted'] = width_weighted
    results['width_baseline'] = width_baseline    
    return results


def test_online(data, algo, list_t, window_sizes_train, window_sizes_infer, weights, test_next = True, alpha = 0.1, delta = 0.1, seed = 2024):
    assert algo == 'RF' or algo == 'XGB'

    num_train = len(window_sizes_train)
    num_infer = len(window_sizes_infer)
    num_weights = len(weights)
    num_t = len(list_t)

    freq_ARW = np.zeros((num_t, num_train))
    width_ARW = np.zeros((num_t, num_train))
    freq_fixed = np.zeros((num_t, num_train, num_infer))
    width_fixed = np.zeros((num_t, num_train, num_infer))
    freq_weighted = np.zeros((num_t, num_train, num_weights))
    width_weighted = np.zeros((num_t, num_train, num_weights))
    width_baseline = np.zeros(num_t)

    for (i, t) in enumerate(list_t):        
        tmp = test(data, algo, t, window_sizes_train, window_sizes_infer, weights, test_next = test_next, alpha = alpha, delta = delta, seed = seed)

        freq_ARW[i] = tmp['freq_ARW']
        freq_fixed[i] = tmp['freq_fixed']
        freq_weighted[i] = tmp['freq_weighted']
        width_ARW[i] = tmp['width_ARW']        
        width_fixed[i] = tmp['width_fixed']
        width_weighted[i] = tmp['width_weighted']
        width_baseline[i] = tmp['width_baseline']
    
    results = dict()
    results['freq_ARW'] = freq_ARW
    results['freq_fixed'] = freq_fixed
    results['freq_weighted'] = freq_weighted
    results['width_ARW'] = width_ARW
    results['width_fixed'] = width_fixed
    results['width_weighted'] = width_weighted
    results['width_baseline'] = width_baseline 
    return results


def summarize(results, alpha = 0.1, idx_start = 0):
    freq_ARW = results['freq_ARW']
    width_ARW = results['width_ARW']
    freq_fixed = results['freq_fixed']
    width_fixed = results['width_fixed']
    freq_weighted = results['freq_weighted']
    width_weighted = results['width_weighted']
    width_baseline = results['width_baseline']

    
    num_models = freq_ARW.shape[1]

    # MAE
    MAE = []
    for j in range(num_models):
        tmp = dict()

        # ARW
        tmp['ARW'] = np.mean(np.abs(freq_ARW[idx_start:, j] - (1 - alpha)))

        # fixed window
        K = freq_fixed.shape[2]
        tmp1 = np.zeros(K)
        for k in range(K):
            tmp1[k] = np.mean(np.abs(freq_fixed[idx_start:, j, k] - (1 - alpha)))
        tmp['fixed'] = tmp1

        # weighted
        K = freq_weighted.shape[2]
        tmp1 = np.zeros(K)
        for k in range(K):
            tmp1[k] = np.mean(np.abs(freq_weighted[idx_start:, j, k] - (1 - alpha)))
        tmp['weighted'] = tmp1

        MAE.append(tmp)

    # average coverage
    AC = []
    for j in range(num_models):
        tmp = dict()

        # ARW
        tmp['ARW'] = np.mean(freq_ARW[idx_start:, j])

        # fixed window
        K = freq_fixed.shape[2]
        tmp1 = np.zeros(K)
        for k in range(K):
            tmp1[k] = np.mean(freq_fixed[idx_start:, j, k])
        tmp['fixed'] = tmp1

        # weighted
        K = freq_weighted.shape[2]
        tmp1 = np.zeros(K)
        for k in range(K):
            tmp1[k] = np.mean(freq_weighted[idx_start:, j, k])
        tmp['weighted'] = tmp1

        AC.append(tmp)

    # average width
    AW = []
    for j in range(num_models):
        tmp = dict()
        
        # Baseline
        tmp['baseline'] = np.mean(width_baseline[idx_start:])

        # ARW
        tmp['ARW'] = np.mean(width_ARW[idx_start:, j])

        # fixed window
        K = freq_fixed.shape[2]
        tmp1 = np.zeros(K)
        for k in range(K):
            tmp1[k] = np.mean(width_fixed[idx_start:, j, k])
        tmp['fixed'] = tmp1

        # weighted
        K = freq_weighted.shape[2]
        tmp1 = np.zeros(K)
        for k in range(K):
            tmp1[k] = np.mean(width_weighted[idx_start:, j, k])
        tmp['weighted'] = tmp1

        AW.append(tmp)

    return MAE, AC, AW



