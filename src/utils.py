"""
Helper functions
"""
import numpy as np

def fit_power_law(x, y):
    # Linear regression in log-log space
    # returns p, a in y = a*x^p
    Y = np.log(y)
    X = np.ones((len(x), 2))
    X[:,0] = np.log(x)
    beta = np.linalg.inv(X.T@X)@X.T@Y
    p, a = beta[0], np.exp(beta[1])
    return p, a

def scree_plot(reliable_variance, all_variance, rank = None, title = None):
    # Helper function for plotting the spectrum
    fig, ax = plt.subplots(1, 2, figsize=(10,4.2))
    ax[0].semilogx(reliable_variance / all_variance * 100, color=GREEN)
    ax[0].set_ylabel("Reliable var")
    ax[1].loglog(reliable_variance / np.sum(reliable_variance), color=GREEN)
    ax[1].set_ylabel("% of reliable var")

    for i, y in enumerate([30, 1]):
        ax[i].set_xlabel("SVC dimension")
        ax[i].set_title(title)
        if rank is not None:
            ax[i].scatter(rank-1, y, color=sns.color_palette()[3], marker='v')

    sns.despine()
    fig.tight_layout()
    return fig, ax
