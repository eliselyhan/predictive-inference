from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

def create_empty_dict(fixed_windows, num_trials):
    d = {}
    d['ARW'] = {}
    for k in fixed_windows:
        d[f'Val_{k}'] = {}
        for trial in range(num_trials):
            d[f'Val_{k}'][trial] = []
            d['ARW'][trial] = []
    return d

def plot_over_time(colors, methods, start_period, end_period, array, methods_to_plot, title, alpha=False):
    fig, ax = plt.subplots(1, 1, figsize = (6,3))
    for method in methods_to_plot:
        plt.plot(array[method], color=colors[method], label=methods[method])
    if alpha:
        plt.axhline(y=1-alpha, color='k', linestyle='--', label=r'nominal')
        #plt.ylabel(r'$F_t(\widehat{q}_{t,\hat{k}})$')
    #if interval: 
        #plt.ylabel(r'$2\widehat{q}_{t,\hat{k}}$')
    plt.title(title)
    plt.xlim(start_period, end_period)
    mpl.rcParams.update({'font.size': 12, 'legend.fontsize': 10})    
    plt.tight_layout()
    plt.xlabel('Time')
    plt.legend()


def summarize_cdf_dict(methods, num_trials, num_periods, cdf_dict, alpha):
    cdf_array = np.zeros((len(methods), num_trials, num_periods))
    for (i, key) in enumerate(cdf_dict.keys()):
        for (trial, trial_coverage) in cdf_dict[key].items():
            cdf_array[i, trial, :] = trial_coverage

    mae_array = np.abs(cdf_array - 1 + alpha)
    mse_array = (cdf_array - 1 + alpha)**2

    #take average of coverage over time and trials for each method
    mae = np.mean(np.mean(mae_array[:, :, 100:], axis=2), axis=1)
    se_ae = np.std(np.mean(mae_array[:, :, 100:], axis=2), axis=1)/np.sqrt(num_trials)

    mse = np.mean(np.mean(mse_array[:, :, 100:], axis=2), axis=1)
    se_mse = np.std(np.mean(mse_array[:, :, 100:], axis=2), axis=1)/np.sqrt(num_trials)
    #barplot errors
    colors = ['r', '#FFA500', 'tab:purple', 'tab:brown', 'tab:green', '#0096FF', 'tab:gray']
    fig, ax = plt.subplots()
    plt.title('Mean Absolute Error')
    for i in range(len(methods)):
        ax.bar(methods[i], mae[i], color=colors[i])

    fig, ax = plt.subplots()
    plt.title('Mean Squared Error')
    for i in range(len(methods)):
        ax.bar(methods[i], mae[i], color=colors[i])

    return cdf_array, mae_array, mae, se_ae, mse_array, mse, se_mse

def summarize_interval_dict(methods, num_trials, num_periods, interval_dict):
    interval_array = np.zeros((len(methods), num_trials, num_periods))
    for (i, key) in enumerate(interval_dict.keys()):
        for (trial, interval) in interval_dict[key].items():
            interval_array[i, trial, :] = interval

    #take average of interval over time and trials for each method
    mean_interval = 2 * np.mean(np.mean(interval_array[:, :, 100:], axis=2), axis=1)
    se_interval = np.std(2 * np.mean(interval_array[:, :, 100:], axis=2), axis=1)/np.sqrt(num_trials)
    #barplot errors
    fig, ax = plt.subplots()
    plt.title('Mean Interval Length')
    colors = ['r', '#FFA500', 'tab:purple', 'tab:brown', 'tab:green', '#0096FF', 'tab:gray']
    for i in range(len(methods)):
        ax.bar(methods[i], mean_interval[i], color=colors[i])
    return interval_array, mean_interval, se_interval