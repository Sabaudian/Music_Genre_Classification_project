import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from itertools import cycle

# my import functions
import constants as const


def makedir(dir_path):
    # create a new directory
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def plot_correlation_matrix(input_data, show_on_screen=True, store_in_folder=True):
    correlation_value = 0.9
    correlation_matrix = input_data.corr(method="pearson", min_periods=40)
    correlated_features = set()

    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) >= correlation_value:
                correlated_features.add(correlation_matrix.columns[i])

    if show_on_screen:
        plt.figure(figsize=(16, 8.5))
        sns.set(font_scale=0.45)
        sns.heatmap(correlation_matrix,
                    cmap="coolwarm",
                    annot=True,
                    fmt=".2g",
                    annot_kws={"size": 5},
                    xticklabels=input_data.keys(),
                    yticklabels=input_data.keys())
        plt.title("Correlation between features", fontsize=18)

        if store_in_folder:
            makedir(const.PLOT_FOLDER)
            plt.savefig(const.PLOT_FOLDER + "/" + const.CORR_MATR_TAG + const.JPG, dpi=300)
        plt.show()


def plot_pca_opt_num_of_components(input_data, cumulative_evr, show_on_screen=True, store_in_folder=True):

    if show_on_screen:

        plt.figure(figsize=(16, 8))
        plt.plot(range(1, len(input_data.columns) + 1), cumulative_evr, marker="o", linestyle="--")
        plt.axhline(y=const.VARIANCE_RATIO, color="red", linestyle="-")
        plt.text(24, 0.81, "80% cut-off threshold", color="red", fontsize=16)
        plt.xticks(range(1, len(input_data.columns) + 1), fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel("Number of Components", fontsize=18)
        plt.ylabel("Cumulative Explained Variance (%)", fontsize=18)
        plt.title("The number of components needed to explain variance", fontsize=22)
        plt.grid()

        if store_in_folder:
            makedir(const.PLOT_FOLDER + "/" + const.CLUSTERING_PLOT_FOLDER)
            plt.savefig(const.PLOT_FOLDER + "/" + const.CLUSTERING_PLOT_FOLDER + "/" + const.OPT_N_COMP_TAG + const.JPG,
                        dpi=300)
        plt.show()


def plot_clusters(input_pca_data, centroids, labels, colors_list, genres_list, show_on_screen=True,
                  store_in_folder=True):
    pca_1, pca_2, genre = input_pca_data["PC1"], input_pca_data["PC2"], input_pca_data["genre"]

    colors = {v: k for v, k in enumerate(colors_list)}
    genres = {v: k for v, k in enumerate(genres_list)}

    df = pd.DataFrame({"pca_1": pca_1, "pca_2": pca_2, "label": labels, "genre": genre})
    groups = df.groupby("label")

    if show_on_screen:

        plt.style.use("ggplot")  # style
        fig, ax = plt.subplots(figsize=(16, 8))

        for genre, group in groups:
            plt.scatter(x=group.pca_1, y=group.pca_2, label=genres[genre], color=colors[genre], edgecolors="white",
                        alpha=0.6)
            ax.tick_params(axis="x", which="both", bottom="off", top="off", labelbottom="off")
            ax.tick_params(axis="y", which="both", left="off", top="off", labelleft="off")

        plt.plot(centroids[:, 0], centroids[:, 1], "*", label="Centroids", c="white", markersize=15, markeredgewidth=1,
                 markeredgecolor="black")

        ax.legend(title="Genres:", fontsize=10)
        ax.set_title("PCA K-Means Clustering", fontsize=22)
        plt.xlabel("PC1", fontsize=16)
        plt.ylabel("PC2", fontsize=16)

        if store_in_folder:
            makedir(const.PLOT_FOLDER + "/" + const.CLUSTERING_PLOT_FOLDER)
            plt.savefig(const.PLOT_FOLDER + "/" + const.CLUSTERING_PLOT_FOLDER + "/" + const.K_MEAN_PCA_CC_TAG +
                        const.JPG, dpi=300)
        plt.show()


def plot_kmeans_confusion_matrix(data, labels, genre_list, show_on_screen=True, store_in_folder=True):
    data["predicted_label"] = labels
    conf_matrix_data = metrics.confusion_matrix(data["genre"], data["predicted_label"])
    conf_matrix = pd.DataFrame(conf_matrix_data, columns=np.unique(genre_list), index=np.unique(genre_list))

    if show_on_screen:
        plt.figure(figsize=(16, 8))
        ax = sns.heatmap(conf_matrix,
                         cmap="Blues",
                         annot=True,
                         fmt="g",
                         annot_kws={"size": 10},
                         square=True,
                         xticklabels=genre_list,
                         yticklabels=genre_list)
        ax.tick_params(labelsize=10)
        plt.xlabel("Predicted Labels", fontsize=16)
        plt.ylabel("True Labels", fontsize=16)
        plt.title("Confusion Matrix for K-Means", fontsize=22)

        if store_in_folder:
            makedir(const.PLOT_FOLDER + "/" + const.CLUSTERING_PLOT_FOLDER)
            plt.savefig(
                const.PLOT_FOLDER + "/" + const.CLUSTERING_PLOT_FOLDER + "/" + const.K_MEAN_CONF_MATR_TAG + const.JPG,
                dpi=300)
        plt.show()


def plot_confusion_matrix(model, model_name, X_train, y_train, X_test, y_test,
                          show_on_screen=True, store_in_folder=True):
    # Fit the model
    model.fit(X_train, y_train)
    # Predict the target vector
    predicts = model.predict(X_test)

    if show_on_screen:
        # Plot confusion matrix
        conf_matrix = confusion_matrix(y_test, predicts)
        plt.figure(figsize=(16, 8))
        ax = sns.heatmap(conf_matrix,
                         cmap="Blues",
                         annot=True,
                         fmt="g",
                         annot_kws={"size": 10},
                         square=True,
                         xticklabels=const.GENRES_LIST,
                         yticklabels=const.GENRES_LIST)
        ax.tick_params(labelsize=10)
        plt.xlabel("Predicted Labels", fontsize=16)
        plt.ylabel("True Labels", fontsize=16)
        plt.title("Confusion Matrix - {}".format(model_name), fontsize=22)

        # save plot into folder
        if store_in_folder:
            makedir(const.PLOT_FOLDER + "/" + const.CLASSIFICATION_PLOT_FOLDER)
            plt.savefig(const.PLOT_FOLDER + "/" + const.CLASSIFICATION_PLOT_FOLDER + "/" + model_name +
                        const.CONF_MATR_TAG + const.JPG)
        plt.show()


def plot_roc(y_test, y_score, operation_name, genres_list, type_of_learning="SL",
             show_on_screen=True, store_in_folder=True):
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
    auc_score = dict()

    for i in range(n_classes):
        false_positive_rate[i], true_positive_rate[i], _ = metrics.roc_curve(test_label[:, i], y_label[:, i])
        auc_score[i] = metrics.auc(false_positive_rate[i], true_positive_rate[i])
    colors = cycle(const.COLORS_LIST)

    if show_on_screen:
        plt.figure(figsize=(16, 8))
        for i, color in zip(range(n_classes), colors):
            plt.plot(false_positive_rate[i], true_positive_rate[i], color=color, lw=1.5,
                     label="ROC curve for {0} (area = {1:0.2f})"
                           "".format(genres[i], auc_score[i]))

        plt.plot([0, 1], [0, 1], "k--", lw=1.5)
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (FPR)", fontsize=24)
        plt.ylabel("True Positive Rate (TPR)", fontsize=24)
        plt.title("Receiver Operating Characteristic Curve for " + operation_name.replace("_", "").upper(), fontsize=24)
        plt.legend(loc="lower right", prop={"size": 12})

        if store_in_folder:

            # plot roc curve for classification methods
            if type_of_learning == "SL":
                makedir(const.PLOT_FOLDER + "/" + const.CLASSIFICATION_PLOT_FOLDER)
                plt.savefig(const.PLOT_FOLDER + "/" + const.CLASSIFICATION_PLOT_FOLDER + "/" +
                            operation_name.replace(" ", "_") + const.ROC_CURVE_TAG + const.JPG, dpi=300)
            else:
                # plot roc curve for k-means clustering
                makedir(const.PLOT_FOLDER + "/" + const.CLUSTERING_PLOT_FOLDER)
                plt.savefig(const.PLOT_FOLDER + "/" + const.CLUSTERING_PLOT_FOLDER + "/" +
                            operation_name.replace(" ", "_") + const.ROC_CURVE_TAG + const.JPG, dpi=300)
        plt.show()


def plot_comparison_of_predictions_by_genre(y_test, y_pred, genres_list, model_name, show_on_screen=True,
                                            store_in_folder=True):
    if show_on_screen:
        compute_confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
        bar = pd.DataFrame(compute_confusion_matrix, columns=genres_list, index=genres_list)
        ax = bar.plot(kind="bar", figsize=(16, 8), fontsize=10, width=0.8, color=const.COLORS_LIST, edgecolor="black")
        ax.legend(loc="upper right", fontsize=8)
        plt.title("Classification Predictions By Genres - " + model_name.upper(), fontsize=18)
        plt.xlabel("Genres", fontsize=14)
        plt.xticks(rotation=0)
        plt.ylabel("Occurrences", fontsize=14)

        for plot in ax.patches:
            if plot.get_height() > 0:
                ax.annotate(format(plot.get_height()) + "%",
                            (plot.get_x() + (plot.get_width() / 2), plot.get_height()), ha="center",
                            va="center", xytext=(0.3, 10), textcoords="offset points", fontsize=5, rotation=90)
        if store_in_folder:
            makedir(const.PLOT_FOLDER + "/" + const.CLASSIFICATION_PLOT_FOLDER)
            plt.savefig(const.PLOT_FOLDER + "/" + const.CLASSIFICATION_PLOT_FOLDER + "/" + model_name.replace(" ", "_")
                        + const.PREDICT_BY_GENRES_TAG + const.JPG,
                        dpi=300)
        plt.show()


def plot_predictions_evaluation(input_data, model_name, genres_list, show_on_screen=True, store_in_folder=True):
    if show_on_screen:

        ax = input_data.plot(kind="bar", figsize=(16, 8), fontsize=14,
                             width=0.6, color=const.PRED_EVA_LIST, edgecolor="black")
        ax.set_xticklabels(genres_list, rotation=0)
        ax.legend(["Real Value", "Predict Value"], fontsize=9, loc="upper right")
        plt.title("Predictions Evaluation - " + model_name.upper(), fontsize=22)
        plt.xlabel("Genres", fontsize=18)
        plt.ylabel("Occurrences", fontsize=18)
        for p in ax.patches:
            ax.annotate(format(p.get_height()),
                        (p.get_x() + (p.get_width() / 2), p.get_height()), ha="center", va="center",
                        xytext=(0, 5), textcoords="offset points", fontsize=10, rotation=0)
        if store_in_folder:
            makedir(const.PLOT_FOLDER + "/" + const.CLASSIFICATION_PLOT_FOLDER)
            plt.savefig(const.PLOT_FOLDER + "/" + const.CLASSIFICATION_PLOT_FOLDER + "/" + model_name +
                        const.PREDICT_EV_TAG + const.JPG, dpi=300)
        plt.show()


def plot_classification_report(clf_report, model_name, show_on_screen=True, store_in_folder=True):
    # exclude support column
    df = clf_report.loc[:, clf_report.columns != "support"]

    if show_on_screen:
        plt.figure(figsize=(16, 8))
        ax = sns.heatmap(df,
                         cmap="RdBu",
                         annot=True,
                         fmt="g",
                         annot_kws={"size": 12},
                         linewidths=1,
                         linecolor="black",
                         cbar=True,
                         clip_on=False)
        ax.tick_params(labelsize=15)
        ax.xaxis.set_ticks_position("top")
        plt.title("{} Classification report".format(model_name), fontsize=22)
        plt.xlabel("Metrics", fontsize=18)
        plt.ylabel("Genres", fontsize=18)

        if store_in_folder:
            makedir(const.PLOT_FOLDER + "/" + const.CLASSIFICATION_PLOT_FOLDER)
            plt.savefig(const.PLOT_FOLDER + "/" + const.CLASSIFICATION_PLOT_FOLDER + "/" + model_name +
                        const.CLF_REPORT_TAG + const.JPG, dpi=300)
        plt.show()
