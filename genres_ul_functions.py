import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# my import functions
import constants as const
import plot_function


def load_data(data_path):
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

    # Scaling
    X_columns = X.columns
    resized_data = preprocessing.MinMaxScaler()
    np_scaled = resized_data.fit_transform(X)

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

    print("\nExplained Variance Ratio for:")
    for i, value in enumerate(pca.explained_variance_ratio_):
        if i <= n_components - 1:
            print("PC{}: \033[92m{}%\033[0m".format(i + 1, round(value * 100, 2)))

    # Plot
    plot_function.plot_pca_opt_num_of_components(input_data=input_data, cumulative_evr=cumulative_evr,
                                                 show_on_screen=show_on_screen, store_in_folder=store_in_folder)
    return n_components


def get_kmeans_model(input_data):
    # K-means model
    kmeans_model = KMeans(n_clusters=10, init="k-means++", n_init="auto", random_state=42).fit(input_data)
    # labels
    kmeans_labels = kmeans_model.labels_
    # centers
    kmeans_centers = kmeans_model.cluster_centers_

    return kmeans_model, kmeans_labels, kmeans_centers


def get_pca_centroids(input_data, input_columns, n_components, centroids):
    column_components = []
    for column in range(n_components):
        column_components.append("PC" + str(column + 1))

    # get PCA components
    pca = PCA(n_components=n_components)
    pca_fit = pca.fit(input_data)
    principal_components = pca_fit.transform(input_data)
    # dataframe
    df = pd.DataFrame(data=principal_components, columns=column_components)

    # concatenate with target label
    pca_data = pd.concat([df.reset_index(drop=True), input_columns.reset_index(drop=True)], axis=1)
    # transform cluster centroids
    pca_centroids = pca_fit.transform(centroids)

    return pca_data, pca_centroids


def silhouette_analysis_for_kmeans_clustering(input_data, min_num_k, max_num_k):
    # list of silhouette values
    silhouette_score_values = list()

    # range of k
    number_of_clusters = range(min_num_k, max_num_k + 1)
    # Compute k-Means with different k
    for k in number_of_clusters:
        clusters = KMeans(n_clusters=k, n_init="auto")
        clusters.fit(input_data)
        cluster_labels = clusters.predict(input_data)
        # append score values in the list
        silhouette_score_values.append(silhouette_score(input_data, cluster_labels,
                                                        metric="euclidean",
                                                        sample_size=None,
                                                        random_state=None))
    # plot function
    plot_function.plot_silhouette(silhouette_score_values=silhouette_score_values,
                                  number_of_clusters=number_of_clusters,
                                  min_num_k=const.MIN_NUM_CLUSTERS,
                                  max_num_k=const.MAX_NUM_CLUSTERS,
                                  show_on_screen=True,
                                  store_in_folder=True)


def k_means_clustering(input_data, input_columns, dataframe, show_cluster, show_confusion_matrix, show_roc_curve,
                       show_silhouette):
    # Number of components
    num_components = number_of_components(input_data=input_data,
                                          variance_ratio=const.VARIANCE_RATIO,
                                          show_on_screen=True,
                                          store_in_folder=True)

    # My K-Means model getting labels and centers
    kmeans_model, labels, centers = get_kmeans_model(input_data)

    # Get PCA and Centroids
    pca, centroids = get_pca_centroids(input_data=input_data.values,
                                       input_columns=input_columns,
                                       n_components=num_components,
                                       centroids=centers)

    if show_cluster:
        # Plot clusters
        plot_function.plot_clusters(input_pca_data=pca[["PC1", "PC2", "genre"]],
                                    centroids=centroids,
                                    labels=labels,
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

    if show_silhouette:
        # Compute and plot silhouette analys on K-Means clustering
        silhouette_analysis_for_kmeans_clustering(input_data=input_data,
                                                  min_num_k=const.MIN_NUM_CLUSTERS,
                                                  max_num_k=const.MAX_NUM_CLUSTERS)


def clustering_and_evaluation(data_path):
    # load normalized data
    X, y, df = load_data(data_path)
    print("\nData:\n\033[92m{}\033[0m".format(df))
    print("\nX (extracted features):\n\033[92m{}\033[0m".format(X))
    print("\ny (genre label):\n\033[92m{}\033[0m".format(y))

    # Plot correlation matrix
    plot_function.plot_correlation_matrix(input_data=X,
                                          show_on_screen=True,
                                          store_in_folder=False)
    # k-means model and evaluation
    k_means_clustering(input_data=X,
                       input_columns=y,
                       dataframe=df,
                       show_cluster=True,
                       show_confusion_matrix=False,
                       show_roc_curve=True,
                       show_silhouette=True)

# # used for testing
# if __name__ == '__main__':
#     # clustering
#     clustering_and_evaluation(data_path=const.DATA_PATH)
