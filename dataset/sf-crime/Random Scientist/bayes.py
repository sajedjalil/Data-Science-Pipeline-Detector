from __future__ import print_function

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Rectangle
import brewer2mpl as b2mpl
import numpy as np
from scipy.stats import itemfreq


def load_dataset():
    df = pd.read_csv('../input/train.csv')

    # dump outliers
    df = df[df.Y < 90]
    return df


def get_map():
    map_extent = [-122.53, 37.68, -122.35, 37.83]
    m = Basemap(llcrnrlon=map_extent[0], llcrnrlat=map_extent[1],
                urcrnrlon=map_extent[2], urcrnrlat=map_extent[3],
                resolution='f', epsg=4269)
    return m


def generate_plots(df, m, bin_size, min_bins):
    # Map-related stuff
    longitudes = np.arange(-122.53, -122.35, .02)
    latitudes = np.arange(37.68, 37.83, .02)

    df = df[df['Category'] != 'OTHER OFFENSES']
    df = df[df['Category'] != 'NON-CRIMINAL']

    # Some relevant quantities.
    categs = pd.Series(sorted(df['Category'].unique()))
    n_categs = len(categs)
    n_samples = 1. * len(df)

    # Posterior mean estimate of category probabilities using Dirichlet prior.
    categ_priors = (df.groupby(['Category']).size() + 1) / (
        n_samples + n_categs)

    # Create the x and y bin edges over which we'll smooth positions.
    # Better approach might be to find optimal bin size through
    # cross validation based approach.
    bin_edges_x = np.arange(np.min(df['X']) - bin_size / 2,
                            np.max(df['X']) + bin_size / 2,
                            bin_size)
    bin_edges_y = np.arange(np.min(df['Y']) - bin_size / 2,
                            np.max(df['Y']) + bin_size / 2,
                            bin_size)

    overall_hist, yedges, xedges = np.histogram2d(
        df.Y, df.X, bins=(bin_edges_y, bin_edges_x))

    # We'll assume that crime really only occurs at a location
    # at which we have seen at least one crime over the 13 years.
    # We'll use this later to make plots look nicer.
    mask = overall_hist == 0
    n_bins = np.sum(overall_hist > 0)

    # Obtain the class condition probabilities p(x|y).
    # In english, we are computing the quantity: given the crime category,
    # what is the probability that the crime occurred in a given xy bin.
    # Because a single crime can happen in only one location, we are
    # treating the class conditional densities as multinomial.
    groups = df.groupby(['Category'])
    px_y = np.zeros([len(bin_edges_y) - 1, len(bin_edges_x) - 1, n_categs])
    px_y_ma = np.ma.masked_where(
        np.tile(np.atleast_3d(mask), [1, 1, n_categs]), px_y)
    for i, (name, group) in enumerate(groups):
        group_hist, yedges, xedges = np.histogram2d(
            group.Y, group.X, bins=(bin_edges_y, bin_edges_x))
        group_hist_ma = np.ma.masked_where(mask, group_hist)

        # Posterior mean estimates of class conditonal probabilities
        # using Dirichlet prior.
        px_y_ma[:, :, i] = (group_hist_ma + 1.0) / (
            np.sum(group_hist_ma) + n_bins)

    # Put the category prior into the right shape for easy broadcasting.
    p_y = np.atleast_3d(categ_priors.as_matrix()).reshape(1, 1, n_categs)
    p_y = np.tile(p_y, [len(bin_edges_y) - 1, len(bin_edges_x) - 1, 1])
    p_y_ma = np.ma.masked_where(
        np.tile(np.atleast_3d(mask), [1, 1, n_categs]), p_y)

    # Obtain the posterior probabilites of each crime category,
    # given that the crime occurred in a known xy location
    py_x_ma = (p_y_ma * px_y_ma) / np.atleast_3d(
        np.sum(p_y_ma * px_y_ma, axis=2))

    # Compute entropy of posterior distribution
    hy = -np.sum(categ_priors * np.log2(categ_priors))
    hy_x = -np.sum(py_x_ma * np.log2(py_x_ma), axis=2)
    entropy_diff = hy - hy_x

    X, Y = np.meshgrid(bin_edges_x, bin_edges_y)

    # Get the crime category that maximized the posterior probability.
    winner = np.argmax(py_x_ma, axis=2)
    winner_ma = np.ma.masked_where(mask, winner)

    # How many time does each crime category prove to be the most likely?
    counts = itemfreq(winner_ma[winner_ma.mask == False])

    # Sort the counts and take only those crime categories which have
    # at least min_pop_bins bins in which they were the winner
    sorted_counts = counts[np.argsort(counts[:, 1])[::-1], :]
    top_counts = sorted_counts[sorted_counts[:, 1] >= min_bins, :]
    top_counts_less = top_counts[1:, :]
    # Plotting stuff...
    n_colors = top_counts_less.shape[0]
    colors = b2mpl.get_map('Set1', 'Qualitative', n_colors).mpl_colors
    recoded = np.zeros(winner.shape)
    for i, categ_num in enumerate(top_counts[:, 0]):
        recoded[winner == categ_num] = i

    plt.figure(figsize=(12, 10))

    winner_ma = np.ma.masked_where(recoded == 0, recoded)
    m.pcolormesh(X, Y, winner_ma, alpha=0.85,
                 cmap=mpl_colors.ListedColormap(colors))

    winner_ma = np.ma.masked_where(
        np.logical_or(mask, winner != 16), winner)
    m.pcolormesh(X, Y, winner_ma, cmap='Greys', alpha=0.75, edgecolor='None')

    legend_labels = [categs[i] for i in top_counts[:, 0]]

    # separate way of doing legend for the most common crime, which is not
    # part of the colormap
    legend_markers = []
    legend_markers.append(
        Rectangle((0, 0), 1, 1, fc='white', ec='white'))

    # now the other crimes
    legend_edgecolors = colors
    legend_facecolors = colors
    for i in range(len(top_counts_less)):
        legend_markers.append(
            Rectangle((0, 0), 1, 1,
                      fc=legend_facecolors[i], ec=legend_edgecolors[i]))
    legend = plt.legend(legend_markers, legend_labels, labelspacing=.075,
                        handlelength=.5, handletextpad=.1,
                        fancybox=True, frameon=1, loc='upper left')
    frame = legend.get_frame()
    frame.set_facecolor('black')

    texts = legend.get_texts()
    texts[0].set_fontsize(10)
    texts[0].set_color('white')
    for t, c in zip(texts[1:], legend_edgecolors):
        t.set_fontsize(10)
        t.set_color(c)

    # m.arcgisimage(service='Canvas/World_Dark_Gray_Base', xpixels=1500)
    m.drawmapboundary(fill_color='aqua')
    m.fillcontinents(color=(0.25, 0.25, 0.25), zorder=0)
    m.drawparallels(latitudes, color='white', labels=[1, 0, 0, 0],
                    dashes=[5, 5], linewidth=.25)
    m.drawmeridians(longitudes, color='white', labels=[0, 0, 0, 1],
                    dashes=[5, 5], linewidth=.25)

    plt.title("Most likely crime given knowledge of location only,\n"
              "taking into account the prior crime probability\n"
              "(excluding NON-CRIMINAL and OTHER OFFENSES)",
              fontsize=16)
    plt.savefig('Class output.png')

    # Entropy difference plot
    plt.figure(figsize=(12, 10))

    entropy_diff_ma = np.ma.masked_where(mask, entropy_diff)
    max_diff = np.max(entropy_diff_ma)
    min_diff = np.min(entropy_diff_ma)
    max_abs = np.max([max_diff, np.abs(min_diff)])
    m.pcolormesh(X, Y, entropy_diff_ma, cmap='RdBu_r', alpha=0.75,
                 edgecolor='None', vmin=-max_abs, vmax=max_abs)
    cbar = plt.colorbar(shrink=.5)
    cbar.solids.set_edgecolor("face")
    plt.draw()

    # m.arcgisimage(service='Canvas/World_Dark_Gray_Base', xpixels=1500)
    m.drawmapboundary(fill_color='aqua')
    m.fillcontinents(color=(0.25, 0.25, 0.25), zorder=0)
    m.drawparallels(latitudes, color='white', labels=[1, 0, 0, 0],
                    dashes=[5, 5], linewidth=.25)
    m.drawmeridians(longitudes, color='white', labels=[0, 0, 0, 1],
                    dashes=[5, 5], linewidth=.25)

    plt.title('Entropy of prior distribution minus entropy of posterior distribution:\n'
              'positive values indicate less uncertainty about crime category '
              'after observing location', fontsize=16)
    plt.savefig('Entropy.png')

df = load_dataset()
map = get_map()
generate_plots(df, map, .002, 5)