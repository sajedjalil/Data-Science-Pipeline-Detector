import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors


def unique_values_bar_plot(data_matrix, data_description):

    # Get the number of unique values for each feature
    all_feats_n_unique_values = []
    for feat_name in data_matrix:
        feat = data_matrix[feat_name]
        all_feats_n_unique_values.append(np.unique(feat).size)

    # Get colors for bars on a logarithmic scale
    colormap = cm.get_cmap('hsv')
    norm = colors.LogNorm(
        vmax=np.max(np.array(all_feats_n_unique_values)),
        vmin=np.min(np.array(all_feats_n_unique_values)))
    bar_colors = []
    for feat_n_uniques in all_feats_n_unique_values:
        bar_colors.append(colormap(norm(feat_n_uniques)))

    # Visualize on a logarithmic scale
    fig, ax = plt.subplots(facecolor='#ffffff', figsize=(15, 10))
    ind = np.arange(len(all_feats_n_unique_values))
    bar_width = 0.8
    bar_plot = ax.bar(left=ind,
                      height=all_feats_n_unique_values,
                      log=True,
                      width=bar_width,
                      color=bar_colors,
                      edgecolor='none')

    # Add some text for labels, title and axes ticks
    ax.set_ylabel("The number of unique values (on a log scale)")
    ax.set_xlabel("Feature")
    title = ("Bar plot of the number of unique values in "
             "each feature: %s data" % (data_description))
    ax.set_title(title)
    ax.set_xticks(ind + bar_width)
    ax.set_xticklabels(data_matrix.columns, rotation=90, ha='right',
                       fontsize=2)
    ax.patch.set_facecolor('#131919')
    ax.set_xlim(0, len(all_feats_n_unique_values))
    # Remove x ticks
    ax.xaxis.set_ticks_position('none')

    # Add text describing the number of unique values on top of each bar
    for rect in bar_plot:
        height = rect.get_height()
        ax.text(x=rect.get_x() + rect.get_width()/2.,
                y=1.05*height,
                s=str(int(height)),
                ha='center',
                va='bottom',
                fontsize=3,
                color='white')
    plt.show()
    file_name = "unique_values_%s.pdf" % data_description
    plt.savefig(file_name, format='pdf')
    file_name = "unique_values_%s.png" % data_description
    plt.savefig(file_name, format='png')

if __name__ == "__main__":
    x_train = pd.read_csv(filepath_or_buffer="../input/train.csv",
                          index_col=0, sep=',')
    x_test = pd.read_csv(filepath_or_buffer="../input/test.csv",
                         index_col=0, sep=',')
    unique_values_bar_plot(x_train, data_description='train')
    unique_values_bar_plot(x_test, data_description='test')
