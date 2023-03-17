import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# my import functions
import constants as const
import plot_function


def load_data(data_path, type_of_normalization):
    # read data from csv
    raw_dataset = pd.read_csv(data_path)
    print("\nRaw Dataset Keys:\n\033[92m{}\033[0m".format(raw_dataset.keys()))

    # drop unnecessary column
    data = raw_dataset.drop(["filename"], axis=1)

    # encode genre label as integer values
    # i.e.: blues = 0, ..., rock = 9
    encoder = preprocessing.OrdinalEncoder()
    data["genre"] = encoder.fit_transform(data[["genre"]])

    # split df into x and y
    label_column = "genre"
    X = data.loc[:, data.columns != label_column]
    y = data.loc[:, label_column]

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

    return X, y, data


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
    kmeans_model = KMeans(n_clusters=10, init="k-means++", n_init="auto").fit(input_data)
    # labels
    kmeans_labels = kmeans_model.labels_
    # centers
    kmeans_centers = kmeans_model.cluster_centers_

    return kmeans_labels, kmeans_centers, kmeans_model


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
          "Components: \033[92m{}\033[0m".format(n_components, pca.explained_variance_ratio_.sum()))

    # concatenate with target label
    pca_data = pd.concat([df.reset_index(drop=True), input_columns.reset_index(drop=True)], axis=1)
    # transform cluster centroids
    pca_centroids = pca_fit.transform(centroids)

    return pca_data, pca_centroids


def k_means_clustering():

    # load normalized data
    X, y, df = load_data(const.DATA_PATH, "min_max")
    print("\nData:\n\033[92m{}\033[0m".format(df))
    print("\nX:\n\033[92m{}\033[0m".format(X))
    print("\ny:\n\033[92m{}\033[0m".format(y))

    # Number of components
    num_components = number_of_components(input_data=X,
                                          variance_ratio=const.VARIANCE_RATIO,
                                          show_on_screen=True,
                                          store_in_folder=False)
    print("\nNumber of Components: \033[92m{}\033[0m".format(num_components))

    # My K-Means model getting label
    labels, centers, kmeans = get_kmeans_model(X)

    # Get PCA and Centroids
    pca, centroids = get_pca_centroids(input_data=X.values,
                                       input_columns=y,
                                       n_components=num_components,
                                       centroids=centers)

    # Plot clusters
    plot_function.plot_clusters(input_pca_data=pca[["PC1", "PC2", "genre"]],
                                centroids=centroids, labels=labels,
                                colors_list=const.COLORS_LIST,
                                genres_list=const.GENRES_LIST,
                                show_on_screen=True,
                                store_in_folder=False)

    # plot confusion matrix
    plot_function.plot_kmeans_confusion_matrix(data=df,
                                               labels=labels,
                                               genre_list=const.GENRES_LIST,
                                               show_on_screen=True,
                                               store_in_folder=False)
    # plot roc curve
    plot_function.plot_roc(y_test=y.values,
                           y_score=labels,
                           operation_name="K-Means",
                           genres_list=const.GENRES_LIST,
                           type_of_learning="UL",
                           show_on_screen=True,
                           store_in_folder=False)


if __name__ == '__main__':
    # clustering
    k_means_clustering()
