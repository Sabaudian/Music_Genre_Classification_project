import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt

import constants as const


def plot_correlation_matrix(correlation_matrix, store_in_folder=True, show_on_screen=True):

    if show_on_screen:

        plt.figure(figsize=(10, 10))
        sns.set(font_scale=0.5)
        sns.heatmap(correlation_matrix, cmap="Blues", square=True)
        plt.title("Correlation between different features", fontsize=16)

        if store_in_folder:
            plt.savefig("plot/correlation_matrix.jpg")
        plt.show()


def plot_pca(input_pca_data, genre_name, store_in_folder):

    data = input_pca_data.copy()

    genres = {i: genre_name[i] for i in range(0, len(genre_name))}
    data.genre = [genres[int(item)] for item in data.genre]

    plt.figure(figsize=(10, 6))

    sns.scatterplot(x="PC1", y="PC2", data=data, hue="genre", alpha=0.5,
                    palette=const.COLORS_LIST, s=50, edgecolors="black")

    plt.title("PCA on Genres", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=10)
    plt.xlabel("Principal Component 1", fontsize=14)
    plt.ylabel("Principal Component 2", fontsize=14)

    if store_in_folder:
        plt.savefig("plot/pca_scatter_plot.jpg")
    plt.show()


def plot_cluster_and_centroid(input_pca_data, centroids, labels, colors_list, genres_list, store_in_folder=True):

    pca_1, pca_2, genre = input_pca_data["PC1"], input_pca_data["PC2"], input_pca_data["genre"]

    colors = {v: k for v, k in enumerate(colors_list)}
    genres = {v: k for v, k in enumerate(genres_list)}

    df = pd.DataFrame({"pca_1": pca_1, "pca_2": pca_2, "label": labels, "genre": genre})
    groups = df.groupby('label')

    fig, ax = plt.subplots(figsize=(10, 6))

    for genre, group in groups:

        plt.scatter(group.pca_1, group.pca_2, label=genres[genre], color=colors[genre], edgecolors="black", alpha=0.6)
        ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        ax.tick_params(axis='y', which='both', left='off', top='off', labelleft='off')

    plt.plot(centroids[:, 0], centroids[:, 1], "kx", ms=15, mec="white", mew=2)

    ax.legend(title="Genres:")
    ax.set_title("Genres Music Clusters Results", fontsize=16)
    if store_in_folder:
        plt.savefig("plot/pca_kmean_cluster_centroids_plot")
    plt.show()


# def plot_confusion_matrix_k_means(input_data, store_in_folder=True, labels=None, target_names=None):
#     if target_names is None:
#         target_names = []
#     if labels is None:
#         labels = []
#     input_data['predicted_label'] = labels
#     data = metrics.confusion_matrix(input_data['genre'], input_data['predicted_label'])
#
#     df_cm = pd.DataFrame(data, columns=np.unique(target_names), index=np.unique(target_names))
#     df_cm.index.name = 'Actual'
#     df_cm.columns.name = 'Predicted'
#
#     plt.figure(figsize=(10, 10))
#     sns.set(font_scale=1.4)
#     heatmap = sns.heatmap(df_cm, cmap="Blues", annot=True, fmt='g', annot_kws={"size": 8}, square=True)
#     heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45)
#     plt.title('Confusion Matrix for K-Means', fontsize=16)
#     if store_in_folder:
#         plt.savefig("kmeans_confusion_matrix_plot.jpg")
#     plt.show()
