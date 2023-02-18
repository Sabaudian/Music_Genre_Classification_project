import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import constants as const


def plot_correlation_matrix():
    return


def plot_2d_pca(input_pca_data, genre_name, store_in_folder):

    data = input_pca_data.copy()

    genres = {i: genre_name[i] for i in range(0, len(genre_name))}
    data.genre = [genres[int(item)] for item in data.genre]

    plt.figure(figsize=(10, 5))

    sns.scatterplot(x="PC1", y="PC2", data=data, hue='genre', alpha=0.3,
                    palette=const.COLORS_LIST, s=100)

    plt.title("PCA on Genres", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=10)
    plt.xlabel("Principal Component 1", fontsize=14)
    plt.ylabel("Principal Component 2", fontsize=14)

    if store_in_folder:
        plt.savefig("plot/pca_2D_scatter_plot.jpg")
    plt.show()


def plot_3d_pca(input_pca_data, store_in_folder):

    # initialize figure and 3d projection for the PC3 data
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    x_ax = input_pca_data["PC1"]
    y_ax = input_pca_data["PC2"]
    z_ax = input_pca_data["PC3"]

    plot = ax.scatter(x_ax, y_ax, z_ax, c=input_pca_data["genre"], cmap="magma", depthshade=True)

    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="z", labelsize=10)
    ax.set_xlabel("PC1", labelpad=10)
    ax.set_ylabel("PC2", labelpad=10)
    ax.set_zlabel("PC3", labelpad=10)

    fig.colorbar(plot, shrink=0.5, aspect=9)

    if store_in_folder:
        plt.savefig("plot/pca_3D_scatter_plot.jpg")
    plt.show()


def plot_cluster_and_centroid(input_pca_data, centroids=None, labels=None, colors_list=None, genres_list=None, store_in_folder=True):

    if genres_list is None:
        genres_list = []
    if colors_list is None:
        colors_list = []
    if labels is None:
        labels = []
    if centroids is None:
        centroids = []
    pca_1, pca_2, gen = input_pca_data['PC1'], input_pca_data['PC2'], input_pca_data['genre']

    colors = {v: k for v, k in enumerate(colors_list)}
    genres = {v: k for v, k in enumerate(genres_list)}

    df = pd.DataFrame({'pca_1': pca_1, 'pca_2': pca_2, 'label': labels, 'genre': gen})
    groups = df.groupby('label')

    fig, ax = plt.subplots(figsize=(20, 13))

    plt.style.use('fivethirtyeight')
    markers = ['s', 'o', 'v', '<', '>', 'P', '*', 'h', 'd', '8']
    for genre, group in groups:
        ax.plot(group.pca_1, group.pca_2, marker=markers[genre], linestyle='', ms=10, color=colors[genre],
                label=genres[genre], mec='none', alpha=0.5)
        ax.set_aspect('auto')
        ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        ax.tick_params(axis='y', which='both', left='off', top='off', labelleft='off')

        plt.plot(centroids[:, 0], centroids[:, 1], 'kx', ms=14)

    ax.legend()
    ax.set_title("Genres Music Clusters Results", fontsize=18)
    if store_in_folder:
        plt.savefig("plot/cluster_centroids_plot")
    plt.show()

    return
