import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression

def principal_component_analysis(x_train):

    """
    Using PLS instead of PCA.
    """

    # Extract the variable to be predicted
    y_train = x_train["TARGET"]
    x_train = x_train.drop(labels="TARGET", axis=1)
    classes = np.sort(np.unique(y_train))
    labels = ["Satisfied customers", "Unsatisfied customers"]

    # PLS
    x_train_normalized = StandardScaler().fit_transform(x_train)
    pls = PLSRegression(n_components=2)
    x_train_projected = pls.fit_transform(x_train_normalized, y_train)[0]

    # Visualize
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(1, 1, 1)
    colors = [(0.0, 0.63, 0.69), 'black']
    markers = ["o", "D"]
    for class_ix, marker, color, label in zip(
            classes, markers, colors, labels):
        ax.scatter(x_train_projected[np.where(y_train == class_ix), 0],
                  x_train_projected[np.where(y_train == class_ix), 1],
                  marker=marker, color=color, edgecolor='whitesmoke',
                  linewidth='1', alpha=0.9, label=label)
        ax.legend(loc='best')
    plt.title(
        "Scatter plot of the training data examples projected on the "
        "2 first principal components")
    plt.xlabel("Principal axis 1")
    plt.ylabel("Principal axis 2")
    plt.show()

    plt.savefig("pca.pdf", format='pdf')
    plt.savefig("pca.png", format='png')

if __name__ == "__main__":
    x_train = pd.read_csv(filepath_or_buffer="../input/train.csv",
                          index_col=0, sep=',')
    principal_component_analysis(x_train)
