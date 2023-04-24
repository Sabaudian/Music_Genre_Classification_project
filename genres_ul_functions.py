import os.path

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# my import functions
import constants as const
import plot_function


def load_data(data_path, type_of_normalization):
    # read file and drop unnecessary column
    raw_dataset = pd.read_csv(data_path)
    print("\nRaw Dataset Keys:\n\033[92m{}\033[0m".format(raw_dataset.keys()))
    df = raw_dataset.drop(["filename"], axis=1)
    print("\nData Shape: \033[92m{}\033[0m".format(df.shape))

    # encode genre label as integer values
    # i.e.: blues = 0, ..., rock = 9
    encoder = preprocessing.OrdinalEncoder()
    df["genre"] = encoder.fit_transform(df[["genre"]])

    # split df into x and y
    label_column = "genre"
    X = df.loc[:, df.columns != label_column]
    y = df.loc[:, label_column]

    # normalization
    X_columns = X.columns
    if type_of_normalization == "std":
        resized_data = preprocessing.StandardScaler()
        np_scaled = resized_data.fit_transform(X)
    elif type_of_normalization == "min_max":
        resized_data = preprocessing.MinMaxScaler()
        np_scaled = resized_data.fit_transform(X)
    else:
        np_scaled = X

    X = pd.DataFrame(np_scaled, columns=X_columns)
    y = pd.DataFrame(y).fillna(0).astype(int)

    return X, y, df


def number_of_components(input_data, variance_ratio, show_on_screen=True, store_in_folder=True):
    # PCA
    pca = PCA()
    pca.fit(input_data)
    # explained_variance
    evr = pca.explained_variance_ratio_
    cumulative_evr = np.cumsum(evr)

    n_components = 0
    for i, ratio in enumerate(cumulative_evr):
        if ratio >= variance_ratio:
            n_components = i + 1
            break
    # Plot
    plot_function.plot_pca_opt_num_of_components(input_data=input_data, cumulative_evr=cumulative_evr,
                                                 show_on_screen=show_on_screen, store_in_folder=store_in_folder)
    return n_components


def get_kmeans_model(input_data):
    # Kmeans model
    kmeans_model = KMeans(n_clusters=3, init="k-means++", n_init="auto").fit(input_data)
    # labels
    kmeans_labels = kmeans_model.labels_
    # centers
    kmeans_centers = kmeans_model.cluster_centers_

    return kmeans_model, kmeans_labels, kmeans_centers


def silhouette_index_computation(input_data):
    k_to_test = range(2, 25, 1)
    silhouette_scores = {}

    for k in k_to_test:
        # compute kmeans with k
        kmeans_model_k = KMeans(n_clusters=k, init="k-means++", n_init="auto").fit(input_data)
        # compute k_th label
        kmeans_label_k = kmeans_model_k.labels_
        # compute the k_th score
        score_k = metrics.silhouette_score(input_data, kmeans_label_k)
        # silhouette values
        silhouette_scores[k] = score_k
        print(
            "Tested k-means with k = \033[92m{:d}\033[0m\tSilhouette Score: \033[92m{:5.4f}\033[0m".format(k, score_k))

    plot_function.plot_silhouette(silhouette_scores)


def get_pca_centroids(input_data, input_columns, n_components, centroids):
    column_components = []
    for column in range(n_components):
        column_components.append("PC" + str(column + 1))

    # get PCA components
    pca = PCA(n_components=n_components)
    pca_fit = pca.fit(input_data)
    principal_components = pca_fit.transform(input_data)

    df = pd.DataFrame(data=principal_components, columns=column_components)
    print("\nPCA Variance Ratio For \033[92m{}\033[0m "
          "Components: \033[92m{:.4f}\033[0m\n".format(n_components, pca.explained_variance_ratio_.sum()))

    # concatenate with target label
    pca_data = pd.concat([df.reset_index(drop=True), input_columns.reset_index(drop=True)], axis=1)
    # transform cluster centroids
    pca_centroids = pca_fit.transform(centroids)

    return pca_data, pca_centroids


def k_means_clustering(input_data, input_columns, dataframe, show_cluster, show_confusion_matrix, show_roc_curve):
    # Number of components
    num_components = number_of_components(input_data=input_data,
                                          variance_ratio=const.VARIANCE_RATIO,
                                          show_on_screen=False,
                                          store_in_folder=False)
    print("\nOptimal Number of Components: \033[92m{}\033[0m".format(num_components))

    # My K-Means model getting labels and centers
    kmeans_model, labels, centers = get_kmeans_model(input_data)

    print("\nK-means Model: \033[92m{}\033[0m".format(kmeans_model))

    # Get PCA and Centroids
    pca, centroids = get_pca_centroids(input_data=input_data.values,
                                       input_columns=input_columns,
                                       n_components=num_components,
                                       centroids=centers)
    if show_cluster:
        # Plot clusters
        plot_function.plot_clusters(input_pca_data=pca[["PC1", "PC2", "genre"]],
                                    centroids=centroids, labels=labels,
                                    colors_list=const.COLORS_LIST,
                                    genres_list=const.GENRES_LIST,
                                    show_on_screen=True,
                                    store_in_folder=True)

    if show_confusion_matrix:
        # plot confusion matrix
        plot_function.plot_kmeans_confusion_matrix(data=dataframe,
                                                   labels=labels,
                                                   genre_list=const.GENRES_LIST,
                                                   show_on_screen=True,
                                                   store_in_folder=True)
    if show_roc_curve:
        # plot roc curve
        plot_function.plot_roc(y_test=input_columns.values,
                               y_score=labels,
                               operation_name="K-Means",
                               genres_list=const.GENRES_LIST,
                               type_of_learning="UL",
                               show_on_screen=True,
                               store_in_folder=True)


def clustering_and_evaluation(data_path, normalization_type):
    # load normalized data
    X, y, df = load_data(data_path, normalization_type)
    print("\nData:\n\033[92m{}\033[0m".format(df))
    print("\nX (extracted features):\n\033[92m{}\033[0m".format(X))
    print("\ny (genre label):\n\033[92m{}\033[0m".format(y))

    # Plot the boxplot of the standardized features
    plot_function.plot_boxplot(X)

    # Plot correlation matrix
    if not os.path.exists(const.STORE_PATH + const.CORR_MATR_TAG + const.JPG):
        plot_function.plot_correlation_matrix(input_data=X,
                                              show_on_screen=False,
                                              store_in_folder=False)
    # k-means model and evaluation
    k_means_clustering(input_data=X,
                       input_columns=y,
                       dataframe=df,
                       show_cluster=True,
                       show_confusion_matrix=False,
                       show_roc_curve=False)

    # silhouette score testing
    silhouette_index_computation(input_data=X)


if __name__ == '__main__':
    # clustering
    clustering_and_evaluation(data_path=const.DATA_PATH, normalization_type=const.MIN_MAX)
