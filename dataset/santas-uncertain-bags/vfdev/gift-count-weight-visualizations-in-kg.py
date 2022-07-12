import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib import colors
from matplotlib import pyplot as plt


FONTSIZE_LABEL = 12
FONTSIZE_TICK = 10
KDE_NOTICE = ("Do note that weights cannot really get negative values, "
              "it's just the prolonged estimates of the distributions.")


# Lbs to Kg
LBS_KG = 0.45359237
KG_LBS = 1.0/LBS_KG

GIFT_WEIGHT_DISTRIBUTIONS = {
    'horse': lambda: LBS_KG * max(0, np.random.normal(5, 2, 1)[0]),
    'ball': lambda: LBS_KG * max(0, 1 + np.random.normal(1, 0.3, 1)[0]),
    'bike': lambda: LBS_KG * max(0, np.random.normal(20, 10, 1)[0]),
    'train': lambda: LBS_KG * max(0, np.random.normal(10, 5, 1)[0]),
    'coal': lambda: LBS_KG * 47 * np.random.beta(0.5, 0.5, 1)[0],
    'book': lambda: LBS_KG * np.random.chisquare(2, 1)[0],
    'doll': lambda: LBS_KG * np.random.gamma(5, 1, 1)[0],
    'blocks': lambda: LBS_KG * np.random.triangular(5, 10, 20, 1)[0],
    'gloves': lambda: LBS_KG * (3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3
                       else LBS_KG * np.random.rand(1)[0])}


def simulate_gift_weights(df, n_observations_per_gift=1000):
    # Get unique gift types
    gift_types = df['GiftId'].apply(lambda x: x.split('_')[0]).unique()
    # Draw observations of weights from each gift type weight distributions
    simulated_data = pd.DataFrame()
    for gift_type in gift_types:
        simulated_data[gift_type.title()] = [
            GIFT_WEIGHT_DISTRIBUTIONS[gift_type]() for _ in
            range(n_observations_per_gift)]
    return simulated_data


def visualize_gift_type_counts(df):
    # Get unique gift types and their counts
    gift_types = df['GiftId'].apply(lambda x: x.split('_')[0]).value_counts()
    print(gift_types)
    # Capitalize gift types
    gift_types.index = [gift.title() for gift in gift_types.index]
    # Generate colors for bars
    colormap = cm.get_cmap('Spectral')
    norm = colors.Normalize(vmax=gift_types.max(), vmin=gift_types.min())
    bar_colors = [colormap(norm(val)) for val in gift_types]
    # Visualize
    ax = gift_types.plot.barh(figsize=(10, 7), fontsize=FONTSIZE_TICK,
                              color=bar_colors, alpha=0.8,
                              title="Counts of gift types in the "
                                    "provided data (gifts.csv)")
    ax.set_ylabel("Gift type", fontsize=FONTSIZE_LABEL)
    ax.set_xlabel("Count", fontsize=FONTSIZE_LABEL)
    plt.tight_layout()
    plt.savefig('gift_type_counts.png')


def visualize_gift_type_weight_distributions(df, n_observations_per_gift):
    # Separately
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(10, 10))
    plt.suptitle("KDE and histogram of %s simulated weights separately for "
                 "each gift type\n%s" % (n_observations_per_gift, KDE_NOTICE),
                 fontsize=FONTSIZE_LABEL, y=0.96)
    for i, gift_type in enumerate(df.columns, start=1):
        # Plot distribution of weights for each gift type
        ax = fig.add_subplot(3, 3, i)
        sns.distplot(df[gift_type], ax=ax, kde_kws={'color': '#002F2F'},
                     hist_kws={'color': '#046380'})
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(6)
    plt.savefig('gift_type_weight_distributions_separately.png')
    # Stacked
    # -------------------------------------------------------------------------
    ax = plt.figure(figsize=(10, 7)).add_subplot(111)
    min_x, max_x, max_y = 0, 0, 0
    for gift_type, color in zip(df.columns, sns.color_palette(
            palette="Paired", n_colors=len(df.columns))):
        ax_kde = sns.kdeplot(df[gift_type], ax=ax, linewidth=3, color=color,
                             alpha=0.8)
        min_x_current, max_x_current = ax_kde.get_xlim()
        min_y_current, max_y_current = ax_kde.get_ylim()
        if max_y_current > max_y:
            max_y = max_y_current
        if min_x_current < min_x:
            min_x = min_x_current
        if max_x_current > max_x:
            max_x = max_x_current
    ax.set_xlim((min_x, max_x))
    ax.set_ylim((-0.005, max_y+0.01))
    ax.set_title("Stacked KDEs of %s simulated weights for each gift type.\n"
                 "%s" % (n_observations_per_gift, KDE_NOTICE),
                 fontsize=FONTSIZE_LABEL)
    ax.set_xlabel("Weight (kg)", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Density", fontsize=FONTSIZE_LABEL)
    plt.tight_layout()
    plt.savefig('gift_type_weight_distributions_stacked.png')


def visualize_gift_type_weight_box_plots(df, n_observations_per_gift):
    ax = plt.figure(figsize=(10, 7)).add_subplot(111)
    sns.boxplot(data=df, ax=ax, linewidth=0.7, fliersize=3,
                palette=sns.color_palette(
                    palette="Spectral", n_colors=len(df.columns)))
    mi, ma, std = df.values.min(), df.values.max(), df.values.std()
    height_extend = 0.3
    ax.set_ylim((mi - height_extend * std, ma + height_extend * std))
    ax.set_ylabel("Weight (kg)", fontsize=FONTSIZE_LABEL)
    ax.set_xlabel("Gift type", fontsize=FONTSIZE_LABEL)
    ax.tick_params(axis='x', labelsize=FONTSIZE_TICK)
    ax.tick_params(axis='y', labelsize=FONTSIZE_TICK)
    ax.set_title("Box plots of %s simulated weights for each gift type" %
                 n_observations_per_gift)
    plt.tight_layout()
    plt.savefig('gift_type_weight_box_plots.png')


if __name__ == '__main__':
    np.random.seed(1122345)
    gifts_df = pd.read_csv(os.path.join('..', 'input', 'gifts.csv'))
    n_observations_per_gift = 1000
    simulated_weights = simulate_gift_weights(
        df=gifts_df, n_observations_per_gift=n_observations_per_gift)
    visualize_gift_type_counts(df=gifts_df)
    visualize_gift_type_weight_distributions(
        df=simulated_weights, n_observations_per_gift=n_observations_per_gift)
    visualize_gift_type_weight_box_plots(
        df=simulated_weights, n_observations_per_gift=n_observations_per_gift)
