import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from itertools import cycle

import constants as const


def plot_correlation_matrix(input_data, show_on_screen=True, store_in_folder=True):
    correlation_value = 0.9
    correlation_matrix = input_data.corr(method="pearson", min_periods=40)
    correlated_features = set()

    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) >= correlation_value:
                correlated_features.add(correlation_matrix.columns[i])

    if show_on_screen:
        plt.figure(figsize=(8, 8))
        sns.set(font_scale=0.5)
        sns.heatmap(correlation_matrix, cmap="coolwarm", square=True)
        plt.title("Correlation between different features", fontsize=16)

        if store_in_folder:
            plt.savefig("plot/correlation_matrix.jpg", dpi=300)
        plt.show()


def plot_pca(input_pca_data, genre_list, store_in_folder):
    data = input_pca_data.copy()

    genres = {i: genre_list[i] for i in range(0, len(genre_list))}
    data.genre = [genres[int(item)] for item in data.genre]

    plt.figure(figsize=(10, 6))

    sns.scatterplot(x="PC1", y="PC2", data=data, hue="genre", alpha=0.5,
                    palette=const.COLORS_LIST, s=50, edgecolors="none", linewidth=0)

    plt.title("PCA on Genres", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=10)
    plt.xlabel("Principal Component 1", fontsize=14)
    plt.ylabel("Principal Component 2", fontsize=14)

    if store_in_folder:
        plt.savefig("plot/pca_scatter_plot.jpg", dpi=300)
    plt.show()


def plot_cluster_and_centroid(input_pca_data, centroids, labels, colors_list, genres_list, show_on_screen=True,
                              store_in_folder=True):
    pca_1, pca_2, genre = input_pca_data["PC1"], input_pca_data["PC2"], input_pca_data["genre"]

    colors = {v: k for v, k in enumerate(colors_list)}
    genres = {v: k for v, k in enumerate(genres_list)}

    df = pd.DataFrame({"pca_1": pca_1, "pca_2": pca_2, "label": labels, "genre": genre})
    groups = df.groupby('label')

    if show_on_screen:

        fig, ax = plt.subplots(figsize=(10, 6))

        for genre, group in groups:
            plt.scatter(group.pca_1, group.pca_2, label=genres[genre], color=colors[genre], edgecolors="black",
                        alpha=0.6)
            ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
            ax.tick_params(axis='y', which='both', left='off', top='off', labelleft='off')

        plt.plot(centroids[:, 0], centroids[:, 1], "x", ms=15, mec="white", mew=2)

        ax.legend(title="Genres:")
        ax.set_title("Genres Music Clusters Results", fontsize=16)

        if store_in_folder:
            plt.savefig("plot/pca_k-mean_cluster_centroids_plot", dpi=300)
        plt.show()


def plot_k_mean_confusion_matrix(input_data, labels, genre_list, show_on_screen=True, store_in_folder=True):
    input_data["predicted_label"] = labels
    data = metrics.confusion_matrix(input_data["genre"], input_data["predicted_label"])

    df_cm = pd.DataFrame(data, columns=np.unique(genre_list), index=np.unique(genre_list))
    df_cm.index.name = "Actual"
    df_cm.columns.name = "Predicted"

    if show_on_screen:
        plt.figure(figsize=(8, 8))
        sns.set(font_scale=1)
        heatmap = sns.heatmap(df_cm, cmap="Blues", annot=True, fmt="g", annot_kws={"size": 8}, square=True)
        heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45)
        plt.title("Confusion Matrix for K-Means", fontsize=16)

        if store_in_folder:
            plt.savefig("k-means_confusion_matrix_plot.jpg", dpi=300)
        plt.show()


# # plot confusion matrix for classification models
# def plot_confusion_matrix(models, X_train, y_train, X_test, y_test):
#     for key in models.keys():
#         model_name = key
#         model = models.get(model_name)
#         # Fit the model
#         model.fit(X_train, y_train)
#         # Predict the target vector
#         predicts = models.get(model_name).predict(X_test)
#         # Accuracy
#         print("{} Accuracy: \033[92m{}\033[0m".format(model_name, round(accuracy_score(y_test, predicts), 5)))
#
#         # Plot confusion matrix
#         conf_matrix = confusion_matrix(y_test, predicts)
#         plt.figure(figsize=(16, 8))
#         plt.title("Confusion Matrix - {}".format(model_name), fontsize=16)
#         sns.heatmap(conf_matrix,
#                     cmap="Blues",
#                     annot=True,
#                     xticklabels=const.GENRES_SET,
#                     yticklabels=const.GENRES_SET)
#         plt.show()


def plot_confusion_matrix(model, model_name, X_train, y_train, X_test, y_test, show_on_screen=True,
                          store_in_folder=True):
    # Fit the model
    model.fit(X_train, y_train)
    # Predict the target vector
    predicts = model.predict(X_test)
    # Accuracy
    print("- {} Accuracy: \033[92m{}%\033[0m".format(model_name, round(accuracy_score(y_test, predicts), 5)*100))
    # Precision, recall and f1-scores
    print("\n{}".format(classification_report(y_test, predicts)))

    if show_on_screen:
        # Plot confusion matrix
        conf_matrix = confusion_matrix(y_test, predicts)
        plt.figure(figsize=(16, 8))
        plt.title("Confusion Matrix - {}".format(model_name), fontsize=16)
        sns.heatmap(conf_matrix,
                    cmap="Blues",
                    annot=True,
                    xticklabels=const.GENRES_SET,
                    yticklabels=const.GENRES_SET)
        if store_in_folder:
            # save plot into folder
            plt.savefig("plot/" + model_name + "_confusion_matrix.jpg")
        plt.show()


def plot_roc(y_test, y_score, operation_name, genres_list, type_of_learning="SL", show_on_screen=True, store_in_folder=True):
    genres = genres_list
    ordinal_position = []

    for index in range(0, len(genres_list)):
        ordinal_position.append(index)

    test_label = preprocessing.label_binarize(y_test, classes=ordinal_position)
    if type_of_learning == "SL":
        y_label = y_score
    else:
        y_label = preprocessing.label_binarize(y_score, classes=ordinal_position)

    n_classes = test_label.shape[1]
    false_positive_rate = dict()
    true_positive_rate = dict()
    roc_auc = dict()

    for i in range(n_classes):
        false_positive_rate[i], true_positive_rate[i], _ = metrics.roc_curve(test_label[:, i], y_label[:, i])
        roc_auc[i] = metrics.auc(false_positive_rate[i], true_positive_rate[i])
    colors = cycle({"blue", "red", "green", "darkorange", "chocolate", "lime", "deepskyblue", "silver", "tomato",
                    "purple"})

    if show_on_screen:
        plt.figure(figsize=(16, 8.5))
        for i, color in zip(range(n_classes), colors):
            plt.plot(false_positive_rate[i], true_positive_rate[i], color=color, lw=1.5,
                     label="ROC curve of class {0} (area = {1:0.2f})"
                           "".format(genres[i], roc_auc[i]))

        plt.plot([0, 1], [0, 1], "k--", lw=1.5)
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (FPR)", fontsize=24)
        plt.ylabel("True Positive Rate (TPR)", fontsize=24)
        plt.title("Receiver Operating Characteristic (ROC) for " + operation_name.replace("_", "").upper())
        plt.legend(loc="lower right", prop={"size": 12})

        if store_in_folder:
            plt.savefig("plot/" + operation_name.replace(" ", "_") + "_roc_plot.jpg", dpi=300)
        plt.show()

# # plotting history
# def plot_history(history, model_name, show_on_screen=True, store_in_folder=True):
#
#     if show_on_screen:
#         fig, axs = plt.subplots(2, figsize=(16, 8))
#
#         # create accuracy subplot
#         axs[0].plot(history.history["accuracy"], label="train accuracy")
#         axs[0].plot(history.history["val_accuracy"], label="test accuracy")
#         axs[0].set_ylabel("Accuracy")
#         axs[0].legend(loc="lower right")
#         axs[0].set_title(model_name + " - Accuracy Eval", fontsize=12)
#
#         # create error subplot
#         axs[1].plot(history.history["loss"], label="train error")
#         axs[1].plot(history.history["val_loss"], label="test error")
#         axs[1].set_ylabel("Error")
#         axs[1].set_xlabel("Epoch")
#         axs[1].legend(loc="upper right")
#         axs[1].set_title(model_name + " - Error Eval", fontsize=12)
#
#         if store_in_folder:
#             plt.savefig("plot/" + model_name + "_history_plot.jpg")
#         plt.show()
