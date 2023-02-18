import pandas as pd
import numpy as np
import pylab as pl
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import constants as const
import plot_funtions as plot


def load_data(data_path, normalization):
    # read data from file
    data = pd.read_csv(data_path)
    # dropping unnecessary column
    df = data.drop(["filename"], axis=1)

    ord_enc = preprocessing.OrdinalEncoder()
    df["genre"] = ord_enc.fit_transform(df[["genre"]])

    # split df into x and y
    target_column = "genre"
    x = df.loc[:, df.columns != target_column]
    y = df.loc[:, target_column]

    x_columns = x.columns
    if normalization == "std":
        resized_data = preprocessing.StandardScaler()
        np_scaled = resized_data.fit_transform(x)
    elif normalization == "min_max":
        resized_data = preprocessing.MinMaxScaler()
        np_scaled = resized_data.fit_transform(x)
    else:
        np_scaled = x

    x = pd.DataFrame(np_scaled, columns=x_columns)
    y = pd.DataFrame(y).fillna(0).astype(int)

    return x, y, df


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


def k_means_data_and_model(input_data, clusters_number=1, random_state=10):
    k_means = KMeans(clusters_number, init="k-means++", random_state=random_state, n_init="auto")
    k_means.fit(input_data)
    labels = k_means.labels_
    centroids = k_means.cluster_centers_
    predict_clusters = k_means.predict(input_data)

    return labels, predict_clusters, centroids, k_means


def get_pca_data_and_centroids(input_data, input_columns, num_of_components, centroids_value,
                               show_on_screen=True, type_of_plot="2d"):
    use_data = input_data.copy()
    column_data = input_columns.copy()
    column_components = []
    for column in range(num_of_components):
        column_components.append("PC" + str(column + 1))

    # get PCA components
    pca = PCA(n_components=num_of_components)
    pca_fit = pca.fit(use_data)
    principal_components = pca_fit.transform(use_data)

    principal_df = pd.DataFrame(data=principal_components, columns=column_components)
    print("\nPCA Variance Ratio For {} Components: {}".format(num_of_components, pca.explained_variance_ratio_.sum()))

    # Concatenate With Target Label
    pca_data = pd.concat([principal_df.reset_index(drop=True), column_data.reset_index(drop=True)], axis=1)

    # Transform Clusters Centroids
    pca_centroids = pca_fit.transform(centroids_value)

    # plotting cluster with centroids
    if show_on_screen:
        if type_of_plot == "2d":
            plot.plot_2d_pca(pca_data[["PC1", "PC2", "genre"]], const.GENRES_SET, True)
        elif type_of_plot == "3d":
            plot.plot_3d_pca(pca_data[["PC1", "PC2", "PC3", "genre"]], True)

    return pca_data, pca_centroids


# test main
if __name__ == '__main__':
    x, y, df = load_data(const.DATA_PATH, "min_max")
    print("\nMin-Max Scaler:")
    print("\nFeatures: {}".format(x))
    print("\nGenre: {}".format(y))
    print("\nDataFrame: {}".format(df))

    num_of_components = optimal_number_of_components(x, const.VARIANCE_RATIO, True, True)
    print("\nNumber of Components {}".format(num_of_components))

    # pca_df, variance_ratio = get_pca(x, num_of_components)
    # print("\nGet PCA data:\npca_df:\n {}".format(pca_df))
    # print("\nvariance_ratio:\n {}".format(variance_ratio))

    labels, predict_clusters, centroids, k_means = k_means_data_and_model(x, 10, 20)
    print("\nK-MEANS:\n\nLabels:\n {}".format(labels))
    print("\npPredict_clusters:\n {}".format(predict_clusters))
    print("\nModel:\n {}".format(k_means))

    pca_data, pca_centroids = get_pca_data_and_centroids(x.values, y, num_of_components, centroids, True, "2d")
    print("\nPrincipal Component data:\n\n{}".format(pca_data.head()))

    plot.plot_cluster_and_centroid(pca_data[['PC1', 'PC2', 'genre']], pca_centroids, labels, const.COLORS_LIST, const.GENRES_SET, True)

    # # reading dataset from csv
    # data = pd.read_csv(data_path)
    # data.head()
    #
    # # dropping unnecessary column
    # data.drop(["filename"], axis=1)  # filename column
    # data.head()
    #
    # print("> Number of rows: {}".format(data.shape[0]))
    # print("> Number of columns: {}".format(data.shape[1]))
    #
    # count_features = 0
    # print("\nFeature:")
    # for feature in data.columns:
    #     if feature != "label" and feature != "filename":
    #         count_features += 1
    #         print(" - {}".format(feature))
    # print("\n> Total amount of feature in the dataset: {}".format(count_features))
    #
    # data = data.iloc[0:, 1:]
    # labels = data["label"]
    # features_data = data.loc[:, data.columns != "label"]
    # columns_data = features_data.columns
    #
    # return features_data, labels, columns_data

    ###########

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
