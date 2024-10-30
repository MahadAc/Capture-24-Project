import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.patches as mpatches

def plot_compare(t, y_pred, y_true=None, trace=None, min_trace=0, max_trace=1):
    if y_true is None:
        do_y_true = False
        y_true = y_pred
    else:
        do_y_true = True
    if trace is not None:  # normalize
        if isinstance(trace, (pd.DataFrame, pd.Series)):
            trace = trace.to_numpy()
        trace = (trace - np.min(trace)) / (np.max(trace) - np.min(trace))

    # uniform resampling
    data = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'trace': trace}, index=t).asfreq('30s')
    y_true, y_pred = data[['y_true', 'y_pred']].to_numpy().T.astype('str')
    trace = data['trace'].to_numpy()
    t = data.index.to_numpy()

    LABEL_COLOR = {
        "sleep": "tab:purple",
        "sit-stand": "tab:red",
        "vehicle": "tab:brown",
        "mixed": "tab:orange",
        "walking": "tab:green",
        "bicycling": "tab:olive",
    }

    def ax_plot(ax, t, y, ylabel=None):
        labels = list(LABEL_COLOR.keys())
        colors = list(LABEL_COLOR.values())

        y = max_trace * (y[:, None] == labels)

        ax.stackplot(t, y.T, labels=labels, colors=colors)

        ax.set_ylabel(ylabel)
        ax.set_ylim((min_trace, max_trace))
        ax.set_yticks([])

        ax.xaxis.grid(True, which='major', color='k', alpha=0.5)
        ax.xaxis.grid(True, which='minor', color='k', alpha=0.25)
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y-%m-%d\n%H:%M"))
        ax.xaxis.set_major_locator(mpl.dates.HourLocator(byhour=range(0,24,4)))
        ax.xaxis.set_minor_locator(mpl.dates.HourLocator())

        ax.tick_params(labelbottom=False, labeltop=True, labelsize=8)
        ax.set_facecolor('#d3d3d3')

        ax.plot(t, trace, c='k')

    if do_y_true:
        fig, axs = plt.subplots(nrows=3, figsize=(10, 3))
        ax_plot(axs[0], t, y_true, ylabel='true')
        ax_plot(axs[1], t, y_pred, ylabel='pred')
        axs[1].set_xticklabels([])  # hide ticks for second row
    else:
        fig, axs = plt.subplots(nrows=2, figsize=(10, 3))
        ax_plot(axs[0], t, y_pred, ylabel='pred')

    # legends
    axs[-1].axis('off')
    legend_patches = [mpatches.Patch(facecolor=color, label=label)
                      for label, color in LABEL_COLOR.items()]
    axs[-1].legend(handles=legend_patches,
                   bbox_to_anchor=(0., 0., 1., 1.),
                   ncol=3,
                   loc='center',
                   mode="best",
                   borderaxespad=0,
                   framealpha=0.6,
                   frameon=True,
                   fancybox=True)

    return fig, axs

