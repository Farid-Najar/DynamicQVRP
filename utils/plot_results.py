import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common import results_plotter
import pandas as pd
import seaborn as sns

sns.set_theme('paper', 'whitegrid')

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    rolled = pd.Series(values).rolling(window)
    std = np.array(rolled.std())#/np.sqrt(window)
    mean = np.array(rolled.mean())
    # weights = np.repeat(1.0, window) / window
    return mean, std#np.convolve(values, weights, "valid"), std


def plot_results(log_folder, 
                 title="Learning Curve", 
                 label = None,
                 window = 75,
                 ylim = None,
                 n = 0,
                 ):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    if n:
        ys = []
        for i in range(n):
            x, y = ts2xy(load_results(log_folder[:-1]+str(i)), "timesteps")
            ys.append(y[:10_000])
            print(len(y))
        y = np.mean(ys, axis=0)
    else:
        x, y = ts2xy(load_results(log_folder), "timesteps")
    y, std = moving_average(y, window=window)
    # Truncate x
    x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    if label is None:
        label = 'mean rewards $\pm 2\sigma$'
    plt.plot(x, y, label=label)
    plt.fill_between(x, y - 2*std, y + 2*std, alpha=0.2)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.gca().ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    if ylim is not None:
        plt.ylim(ylim)
    plt.title(title + " Smoothed")
    # plt.legend()
    # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    return x
    # plt.show()
   
   
def plot_curves(res, label, eval_every=200):
    """
    Plot learning curves from evaluation results.

    Parameters
    ----------
    res : numpy.ndarray
        Array containing evaluation results, with shape (num_evaluations, num_episodes)
    label : str
        Label for the plot legend
    eval_every : int, optional
        Number of steps between evaluations, by default 200

    Returns
    -------
    None
        Plots the learning curves with mean and confidence intervals
    """
    y = np.mean(res, 1)
    
    std = np.std(res, 1)/np.sqrt(len(res[0]))
    plt.plot(eval_every*np.arange(len(y)), y, label=label)
    plt.fill_between(
        eval_every*np.arange(len(y)), 
        y - 2*std, 
        y + 2*std, 
        alpha=0.2,
        # label = "95% ci"
    )
     
    
def add_value_labels(ax, spacing=5):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    spacing = min(5, spacing)
    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = "{:.3f}".format(y_value)

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.

def addlabels(x, y, add=True):
    """
    Add labels to the bars in a bar plot.

    Parameters
    ----------
    x : list
        List of x positions.
    y : list
        List of y values.
    add : bool, optional
        Whether to add a small offset to the labels, by default True.
    """
    for i in range(len(x)):
        add_val = np.sign(y[i]) * 1.2 if add else 0
        y_pos = y[i] + add_val
        if abs(y[i]) < 0.15 * max(y):  # Label inside bar for very small values
            y_pos = y[i] / 2
            va = 'center'
        else:  # Label above bar for larger values
            va = 'bottom'
            fontweight = 'normal'
        
        # fontweight = 'thick'
        color = 'black'
        plt.text(i, y_pos, f'{y[i]:.2f}', ha='center', va=va, color=color, fontweight=fontweight)

def plot_gap_method(data : dict, method : str):
    """
    Plot the gap between the specified method and other methods.

    Parameters
    ----------
    data : dict
        Dictionary containing the results for different methods.
    method : str
        The method to compare against.
    """
    gap = {
        k : data[k]/data[method] -1
        for k in data.keys() #if k != method
    }

    sns.boxplot(
        gap,
        # gap.values(),
        # tick_labels=list(gap.keys()),
        showmeans=True,
        meanprops={
            'marker':'o',
            'markerfacecolor':'black',
            'markeredgecolor':'black',
            'markersize':'7'
        },
    )
    plt.hlines(0, -0.5, len(gap), colors='red')
    plt.title(f"methods/{method} ratio")
    plt.show()
    
def plot_gap_offline(data : dict):
    """
    Plot the gap between online and offline methods.

    Parameters
    ----------
    data : dict
        Dictionary containing the results for different methods.
    """
    gap = {
        k : 1 - data[k]/data["Offline"]
        for k in data.keys()# if k != "Offline"
    }
    
    mean_gap = {
        k : 1 - np.mean(data[k]/data["Offline"])
        for k in data.keys() if k != "Offline"
    }

    sns.boxplot(
        gap,
        # gap.values(),
        # tick_labels=list(gap.keys()),
        showmeans=True,
        meanprops={
            'marker':'o',
            'markerfacecolor':'black',
            'markeredgecolor':'black',
            'markersize':'7'
        },
    )
    plt.hlines(0, -0.5, len(gap), colors='red')
    # plt.hlines(0, 0.5, len(gap)+.5, colors='red')
    plt.title("online/offline gap")
    plt.show()
    
    ax = sns.barplot(
        mean_gap
        # gap.values(),
        # tick_labels=list(gap.keys()),
    )
    plt.hlines(0, -0.5, len(mean_gap), colors='red')
    # addlabels(list(mean_gap.keys()), np.round(list(mean_gap.values()), 2), False)
    add_value_labels(ax, spacing=max(mean_gap.values())/2)
    # plt.hlines(0, 0.5, len(gap)+.5, colors='red')
    plt.title("mean online/offline gap")
    plt.show()
    
    
def plot_rewards_dist(data : dict):
    """
    Plot the distribution of rewards for different methods.

    Parameters
    ----------
    data : dict
        Dictionary containing the results for different methods.
    """
    sns.boxplot(
        data,
        # qs,
        # tick_labels=list(data.keys()),
        showmeans=True,
        meanprops={
            'marker':'o',
            'markerfacecolor':'black',
            'markeredgecolor':'black',
            'markersize':'7'
        },
    )
    # plt.hlines(1, 0.5, len(qs)+.5, colors='red')
    plt.title("Rewards distribution")
    plt.show()
    
    
def plot_mean_rewards(data : dict):
    """
    Plot the mean rewards for different methods.

    Parameters
    ----------
    data : dict
        Dictionary containing the results for different methods.
    """
    vs = {
        k : data[k].mean()
        for k in data.keys()
    }

    # plt.bar(
    #     list(vs.keys()),
    #     list(vs.values()),

    # )
    
    ax = sns.barplot(
        data
    )
    
    # calling the function to add value labels
    # addlabels(list(vs.keys()), np.round(list(vs.values()), 2))
    add_value_labels(ax, spacing=max(vs.values())/2)
    
    plt.ylim(0, 1.2*max(list(vs.values())))
    
    # plt.hlines(1, 0.5, len(gap)+.5, colors='red')
    plt.title("Mean rewards by methods")
    plt.show()
    # plt.hlines(np.mean(r_MSA_woRO/r_offline_woRO), 0.5, 2.5, colors='red', linestyles='--')
    
def plot_mean_occupancy(data : dict, H = None):
    """
    Plot the mean service rate for different methods.

    Parameters
    ----------
    data : dict
        Dictionary containing the results for different methods.
    """
    if H is None:
        return
    
    vs = {
        k : 100*data[k]/H
        for k in data.keys()
    }

    # plt.bar(
    #     list(vs.keys()),
    #     list(vs.values()),

    # )
    
    ax = sns.barplot(
        vs
    )
    
    # calling the function to add value labels
    # addlabels(list(vs.keys()), np.round(list(vs.values()), 2))
    add_value_labels(ax, spacing=4)#max(vs.values())/2)
    
    # plt.ylim(0, 110)
    plt.ylim(0, 1.1*np.amax(list(vs.values())))
    
    # plt.hlines(1, 0.5, len(gap)+.5, colors='red')
    plt.title("Service/acceptance rate by methods")
    plt.ylabel("Service/acceptance rate in %")
    plt.show()
    
    
def plot_dist_occupancy(data : dict, total_cap = None):
    """
    Plot the distribution of service rate for different methods.

    Parameters
    ----------
    data : dict
        Dictionary containing the results for different methods.
    """
    if total_cap is None:
        return
    
    vs = {
        k : 100*data[k]/total_cap
        for k in data.keys()
    }

    # plt.bar(
    #     list(vs.keys()),
    #     list(vs.values()),

    # )
    
    # ax = sns.barplot(
    #     vs
    # )
    
    ax = sns.boxplot(
        vs,
        # qs,
        # tick_labels=list(data.keys()),
        showmeans=True,
        meanprops={
            'marker':'o',
            'markerfacecolor':'black',
            'markeredgecolor':'black',
            'markersize':'7'
        },
    )
    
    # calling the function to add value labels
    # addlabels(list(vs.keys()), np.round(list(vs.values()), 2))
    # add_value_labels(ax, spacing=4)#max(vs.values())/2)
    
    plt.ylim(0, 110)
    
    # plt.hlines(1, 0.5, len(gap)+.5, colors='red')
    plt.title("Service rate by methods")
    plt.ylabel("Service rate in %")
    plt.show()
    
def plot_improvement(data : dict):
    """
    Plot the improvement percentage of mean rewards compared to the greedy method.

    Parameters
    ----------
    data : dict
        Dictionary containing the results for different methods.
    """
    gap = np.array([
        # res_offline.mean()/res_greedy.mean(),
        data[k].mean()/data["FAFS"].mean()
        for k in data.keys()
    ])

    args = np.argsort(-gap)
    gap = gap[args]

    gap -= 1

    mask_neg = gap < 0
    mask_pos = gap >= 0

    x = np.array(list(data.keys()))

    x = x[args]

    sns.barplot(
        x = x[mask_pos],
        y = 100*gap[mask_pos],
        color='green'
    )

    ax = sns.barplot(
        x = x[mask_neg],
        y = 100*gap[mask_neg],
        color='red'
    )
    
    # calling the function to add value labels
    # addlabels(x, np.round(100*gap, 2))
    add_value_labels(ax, spacing=max(100*gap)/2)
    
    diffmM = max(100*gap) - min(100*gap)
    scale = 1.2*diffmM
    plt.ylim(min(100*gap) - 0.1*scale, max(100*gap) + 0.1*scale)
    # plt.ylim(1.2*min(100*gap), 1.2*max(100*gap))

    # plt.hlines(1, 0.5, len(gap)+.5, colors='red')
    plt.title("Improvement % of mean rewards compared to FAFS")
    plt.ylabel("Improvement in %")
    plt.show()
   
def plot_improvement2(data : dict):
    """
    Plot the mean improvement percentage compared to the greedy method.

    Parameters
    ----------
    data : dict
        Dictionary containing the results for different methods.
    """
    gap = np.array([
        # res_offline.mean()/res_greedy.mean(),
        np.mean(data[k]/data["FAFS"])
        for k in data.keys()
    ])

    args = np.argsort(-gap)
    gap = gap[args]

    gap -= 1

    mask_neg = gap < 0
    mask_pos = gap >= 0

    x = np.array(list(data.keys()))

    x = x[args]

    sns.barplot(
        x = x[mask_pos],
        y = 100*gap[mask_pos],
        color='green',
    )

    ax = sns.barplot(
        x = x[mask_neg],
        y = 100*gap[mask_neg],
        color='red',
        # errorbar=('ci', .95)
    )
    
    # calling the function to add value labels
    # addlabels(x, np.round(100*gap, 2))
    add_value_labels(ax, spacing=max(100*gap)/2)
    
    diffmM = max(100*gap) - min(100*gap)
    scale = 1.2*diffmM
    plt.ylim(min(100*gap) - 0.1*scale, max(100*gap) + 0.1*scale)
    # plt.ylim(1.2*min(100*gap), 1.2*max(100*gap))

    # plt.hlines(1, 0.5, len(gap)+.5, colors='red')
    plt.title("Mean % improvement compared to FAFS")
    plt.ylabel("Improvement in %")
    plt.show() 

def plot(data : dict, total_cap = None, H = None):
    """
    Plot various metrics and comparisons for the given data.

    Parameters
    ----------
    data : dict
        Dictionary containing the results for different methods.
    """
    plot_improvement(data)
    plot_improvement2(data)
    plot_mean_rewards(data)
    plot_mean_occupancy(data, H)
    plot_rewards_dist(data)
    plot_gap_offline(data)
    plot_gap_method(data, 'FAFS')
    # plot_gap_method(data, 'MSA')
    plot_gap_method(data, 'Random')
    plot_gap_method(data, 'DQN \nVA')
    plot_dist_occupancy(data, H)
