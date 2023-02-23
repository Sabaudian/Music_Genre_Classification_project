import numpy as np
import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import constants as const
import plot_function


def load_data(data_path, type_of_normalization):
    # read data dropping unnecessary
    raw_dataset = pd.read_csv(data_path)
    print("\nRaw Dataset Keys:\n\033[92m{}\033[0m".format(raw_dataset.keys()))

    # drop unnecessary column
    data = raw_dataset.drop(["filename"], axis=1)

    # encode genre label as integer values
    # i.e.: blues = 0, ..., rock = 9
    genre_encode = preprocessing.OrdinalEncoder()
    data["genre"] = genre_encode.fit_transform(data[["genre"]])

    # split df into x and y
    label_column = "genre"
    X = data.loc[:, data.columns != label_column]
    y = data.loc[:, label_column]

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


# Find the optimal number of components
def optimal_number_of_components(input_data, variance_ratio, store_in_folder=True, show_on_screen=True):
    pca = PCA()
    pca.fit(input_data)
    exp_var_rat = pca.explained_variance_ratio_
    cum_exp_var_rat = np.cumsum(exp_var_rat)

    number_of_components = 0
    for i, exp_var in enumerate(cum_exp_var_rat):
        if exp_var >= variance_ratio:
            number_of_components = i + 1
            break

    # The plot show the amount of variance captured (on the y-axis) depending on the number of components we include (the x-axis)
    if show_on_screen:
        plt.figure(figsize=(8, 8))
        plt.bar(range(1, len(input_data.columns) + 1), exp_var_rat, alpha=0.75, align="center",
                label="individual explained variance", color="green")
        plt.xticks(np.arange(1, len(input_data.columns) + 1, 1))
        plt.step(range(1, len(input_data.columns) + 1), cum_exp_var_rat,
                 c="red",
                 label="cumulative explained variance")
        plt.ylim(0, 1.1)
        plt.xlabel("Principal Components", fontsize=22)
        plt.ylabel("Explained variance ratio", fontsize=22)
        plt.legend(loc="upper left", prop={"size": 12}, frameon=True, framealpha=1)

    # Store image in the plot folder
    if store_in_folder:
        pl.savefig("plot/pca_cumulative_variance_ratio_plot.jpg")
    plt.show()

    return number_of_components


def get_pca_data_and_centroids(input_data, input_columns, num_of_components, centroids_value, show_on_screen=True):
    column_components = []
    for column in range(num_of_components):
        column_components.append("PC" + str(column + 1))

    # get PCA components
    pca = PCA(n_components=num_of_components)
    pca_fit = pca.fit(input_data)
    principal_components = pca_fit.transform(input_data)

    principal_df = pd.DataFrame(data=principal_components, columns=column_components)
    print("\nPCA Variance Ratio For \033[92m{}\033[0m Components: \033[92m{}\033[0m".format(num_of_components,
                                                                                            pca.explained_variance_ratio_.sum()))

    # concatenate with target label
    pca_data = pd.concat([principal_df.reset_index(drop=True), input_columns.reset_index(drop=True)], axis=1)
    # transform cluster centroids
    pca_centroids = pca_fit.transform(centroids_value)

    # plot PCA
    if show_on_screen:
        plot_function.plot_pca(input_pca_data=pca_data[["PC1", "PC2", "genre"]],
                               genre_name=const.GENRES_SET,
                               store_in_folder=False)

    return pca_data, pca_centroids


def k_means_model(input_data, clusters_number=1, random_state=10):
    k_means = KMeans(clusters_number, random_state=random_state, n_init="auto")
    k_means.fit(input_data)
    labels = k_means.labels_
    centroids = k_means.cluster_centers_
    predict_clusters = k_means.predict(input_data)

    return labels, predict_clusters, centroids, k_means


if __name__ == '__main__':
    # load data and normalize
    X, y, data = load_data(const.DATA_PATH, "min_max")
    print("\nMy Data:\n\033[92m{}\033[0m".format(data))
    print("\nFeatures to Cluster:\n\033[92m{}\033[0m".format(X))
    print("\nLabel:\n\033[92m{}\033[0m".format(y))

    # pick optimal number of components to use
    num_of_components = optimal_number_of_components(input_data=X,
                                                     variance_ratio=0.8,
                                                     store_in_folder=False,
                                                     show_on_screen=False)
    print("\nOptimal Number of Components: \033[92m{}\033[0m".format(num_of_components))

    # compute k-means and extract features from it
    labels, predict_clusters, centroids, k_means = k_means_model(input_data=X,
                                                                 clusters_number=10,
                                                                 random_state=20)

    # get pca and centroids
    pca_data, pca_centroids = get_pca_data_and_centroids(input_data=X.values,
                                                         input_columns=y,
                                                         num_of_components=num_of_components,
                                                         centroids_value=centroids,
                                                         show_on_screen=False)
    print("\nPrincipal Component data:\n\033[92m{}\033[0m".format(pca_data))

    # plot cluster
    plot_function.plot_cluster_and_centroid(input_pca_data=pca_data[["PC1", "PC2", "genre"]],
                                            centroids=pca_centroids,
                                            labels=labels,
                                            colors_list=const.COLORS_LIST,
                                            genres_list=const.GENRES_SET,
                                            store_in_folder=False)

    # scaler = StandardScaler()
    # X = scaler.fit_transform(np.array(data, dtype=float))
    #
    # dat = (X - np.min(X)) / (np.max(X) - np.min(X))
    #
    # pca = PCA().fit(dat)
    # plt.figure()
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.xlabel('Number of Components')
    # plt.ylabel('Variance (%)')  # for each component
    # plt.title(' Dataset Explained Variance')
    # plt.show()
    #
    # # silhouette_score
    # result = []
    # # n_clusters=3
    # for n_clusters in list(range(3, 25)):
    #     clusterer = KMeans(n_clusters=n_clusters, init='k-means++', n_init='auto').fit(X)
    #     preds = clusterer.predict(X)
    #     centers = clusterer.cluster_centers_
    #     result.append(silhouette_score(X, preds, sample_size=26))
    #
    # plt.plot(range(3, 25), result, 'bx-')
    # plt.xlabel('number of clusters')
    # plt.ylabel('result')
    #
    # plt.show()
