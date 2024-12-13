import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common import results_plotter
import pandas as pd

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
    

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, np.sign(y[i])*1.2 + y[i], y[i], ha = 'center')
        
def plot_gap_greedy(data : dict):
    gap = {
        k : data[k]["rs"]/data["Greedy"]["rs"]
        for k in data.keys() if k != "Greedy"
    }

    plt.boxplot(
        gap.values(),
        tick_labels=list(gap.keys()),
    )
    plt.hlines(1, 0.5, len(gap)+.5, colors='red')
    plt.title("methods/greedy ratio")
    plt.show()
    
def plot_gap_offline(data : dict):
    gap = {
        k : data[k]["rs"]/data["Offline"]["rs"]
        for k in data.keys() if k != "Offline"
    }

    plt.boxplot(
        gap.values(),
        tick_labels=list(gap.keys()),
    )
    plt.hlines(1, 0.5, len(gap)+.5, colors='red')
    plt.title("online/offline ratio")
    plt.show()
    
def plot_gap_MSA(data : dict):
    gap = {
        k : data[k]["rs"]/data["MSA"]["rs"]
        for k in data.keys() if k != "MSA"
    }

    plt.boxplot(
        gap.values(),
        tick_labels=list(gap.keys()),
    )
    plt.hlines(1, 0.5, len(gap)+.5, colors='red')
    plt.title("methods/MSA ratio")
    plt.show()
    
def plot_rewards_dist(data : dict):
    qs = [
        res["rs"] for res in data.values()
    ]

    plt.boxplot(
        qs,
        tick_labels=list(data.keys()),

    )
    # plt.hlines(1, 0.5, len(qs)+.5, colors='red')
    plt.title("Rewards distribution")
    plt.show()
    
    
def plot_mean_rewards(data : dict):

    vs = {
        k : data[k]["rs"].mean()
        for k in data.keys()
    }

    plt.bar(
        list(vs.keys()),
        list(vs.values()),

    )
    # plt.hlines(1, 0.5, len(gap)+.5, colors='red')
    plt.title("Mean rewards by methods")
    plt.show()
    # plt.hlines(np.mean(r_MSA_woRO/r_offline_woRO), 0.5, 2.5, colors='red', linestyles='--')
    
    
def plot_improvement(data : dict):

    gap = np.array([
        # res_offline["rs"].mean()/res_greedy["rs"].mean(),
        data[k]["rs"].mean()/data["Greedy"]["rs"].mean()
        for k in data.keys()
    ])

    args = np.argsort(-gap)
    gap = gap[args]

    gap -= 1

    mask_neg = gap < 0
    mask_pos = gap >= 0

    x = np.array(list(data.keys()))

    x = x[args]

    plt.bar(
        x[mask_pos],
        100*gap[mask_pos],
        color='green'
    )

    plt.bar(
        x[mask_neg],
        100*gap[mask_neg],
        color='red'
    )
    
    # calling the function to add value labels
    addlabels(x, np.round(100*gap, 2))
    
    plt.ylim(1.2*min(100*gap), 1.2*max(100*gap))

    # plt.hlines(1, 0.5, len(gap)+.5, colors='red')
    plt.title("Mean % improvement compared to greedy by methods")
    plt.ylabel("Improvement in %")
    plt.show()
    

def plot(data : dict):
    plot_improvement(data)
    plot_mean_rewards(data)
    plot_rewards_dist(data)
    plot_gap_offline(data)
    plot_gap_greedy(data)
    plot_gap_MSA(data)
    