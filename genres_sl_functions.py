import os
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# my import functions
import constants as const
import plot_function


def makedir(dir_path):
    # create a new directory
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


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

    X_columns = X.columns
    resized_data = preprocessing.MinMaxScaler()
    np_scaled = resized_data.fit_transform(X)

    X = pd.DataFrame(np_scaled, columns=X_columns)

    return X, y, df


def prepare_datasets(X, y, test_size):
    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


def get_classification_model():
    # models dictionary
    models = {"NN": [], "RF": [], "KNN": [], "SVM": []}

    # Neural Network
    nn_model = MLPClassifier(solver="adam", alpha=1e-5, hidden_layer_sizes=(16, 16), random_state=1,
                             activation="relu", learning_rate="adaptive", early_stopping=False, verbose=False,
                             max_iter=1000)
    models.update({"NN": nn_model})

    # Random forest
    rf_model = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=10)
    models.update({"RF": rf_model})

    # k-Nearest Neighbors
    knn_model = KNeighborsClassifier(weights="distance")
    models.update({"KNN": knn_model})

    # Support Vector Machine
    svc_model = SVC(C=10, kernel="rbf", probability=True, random_state=10)
    models.update({"SVM": svc_model})

    return models


def compute_evaluation_metrics(model, model_name, X_test, y_test):
    # Predict the target vector
    y_predict = model.predict(X_test)

    # compute report
    clf_report = classification_report(y_test, y_predict, target_names=const.GENRES_LIST, digits=2, output_dict=True)
    # update so in df is shown in the same way as standard print
    clf_report.update({"accuracy": {"precision": None, "recall": None, "f1-score": clf_report["accuracy"],
                                    "support": clf_report.get("macro avg")["support"]}})
    df = pd.DataFrame(clf_report).transpose()

    # save report into file
    makedir(const.DATA_FOLDER + "/" + const.CLF_REPORT_PATH)
    df.to_csv(const.DATA_FOLDER + "/" + const.CLF_REPORT_PATH + "/" + model_name + "_classification_report.csv",
              index=True, float_format="%.5f")
    return df


def prediction_comparison(model, X_test, y_test):
    # Predict the target vector
    y_predict = model.predict(X_test)
    # Genres
    genres = {i: const.GENRES_LIST[i] for i in range(0, len(const.GENRES_LIST))}

    clf_data = pd.DataFrame(columns=["real_genre_num", "predict_genre_num",
                                     "real_genre_text", "predict_genre_text"])
    clf_data["real_genre_num"] = y_test.astype(int)
    clf_data["predict_genre_num"] = y_predict.astype(int)

    # compare real values with predicted values
    comparison_column = np.where(clf_data["real_genre_num"] == clf_data["predict_genre_num"], True, False)
    clf_data["check"] = comparison_column

    clf_data["real_genre_text"] = clf_data["real_genre_num"].replace(genres)
    clf_data["predict_genre_text"] = clf_data["predict_genre_num"].replace(genres)

    input_data = pd.DataFrame()
    input_data[["Genre", "Real_Value"]] = \
        clf_data[["real_genre_text", "predict_genre_text"]].groupby(["real_genre_text"], as_index=False).count()
    input_data[["Genre", "Predict_Value"]] = \
        clf_data[["real_genre_text", "predict_genre_text"]].groupby(["predict_genre_text"], as_index=False).count()

    return input_data


def model_evaluation(models, X_train, y_train, X_test, y_test,
                     show_confusion_matrix=True, show_roc_curve=True,
                     show_compare_prediction_by_genre=True, show_simple_compare=True):
    # evaluation of every classification model
    for key, value in models.items():

        # NN, KNN, RF and SVM
        model_name = key
        # computed model
        model_type = value

        if show_confusion_matrix:
            # plotting confusion matrix
            plot_function.plot_confusion_matrix(model=model_type,
                                                model_name=model_name,
                                                X_train=X_train,
                                                y_train=y_train,
                                                X_test=X_test,
                                                y_test=y_test,
                                                show_on_screen=True,
                                                store_in_folder=True)

        if model_name == "SVM":
            y_score = model_type.fit(X_train, y_train).decision_function(X_test)
        else:
            model_type.fit(X_train, y_train)
            y_score = model_type.predict_proba(X_test)

        if show_roc_curve:
            # Plotting roc curve
            plot_function.plot_roc(y_test=y_test,
                                   y_score=y_score,
                                   operation_name=model_name,
                                   genres_list=const.GENRES_LIST,
                                   type_of_learning="SL",
                                   show_on_screen=True,
                                   store_in_folder=True)

        if show_compare_prediction_by_genre:
            # Predict the target vector
            y_predict = model_type.predict(X_test)
            # plot histogram
            plot_function.plot_comparison_of_predictions_by_genre(y_test=y_test,
                                                                  y_pred=y_predict,
                                                                  genres_list=const.GENRES_LIST,
                                                                  model_name=model_name,
                                                                  show_on_screen=True,
                                                                  store_in_folder=True)
        if show_simple_compare:
            input_data = prediction_comparison(model=model_type, X_test=X_test, y_test=y_test)
            # evaluation actual/prediction
            plot_function.plot_predictions_evaluation(input_data=input_data,
                                                      model_name=model_name,
                                                      genres_list=const.GENRES_LIST,
                                                      show_on_screen=True,
                                                      store_in_folder=True)
        # metrics computation
        clf_report = compute_evaluation_metrics(model=model_type, model_name=model_name, X_test=X_test, y_test=y_test)
        # plot classification report
        plot_function.plot_classification_report(clf_report=clf_report,
                                                 model_name=model_name,
                                                 show_on_screen=True,
                                                 store_in_folder=True)


def classification_and_evaluation(data_path):
    # load data
    X, y, df = load_data(data_path=data_path)
    print("\nData:\n\033[92m{}\033[0m".format(df))
    print("\nX (my data):\n\033[92m{}\033[0m".format(X))
    print("\ny (labels):\n\033[92m{}\033[0m".format(y))

    # Plot correlation matrix
    plot_function.plot_correlation_matrix(input_data=X,
                                          show_on_screen=False,
                                          store_in_folder=False)

    # create train/test split
    X_train, X_test, y_train, y_test = prepare_datasets(X=X, y=y, test_size=0.3)
    print("\nSplit data into Train and Test:")
    print("- Train set has \033[92m{}\033[0m"
          " records out of \033[92m{}\033[0m"
          " which is \033[92m{}%\033[0m".format(X_train.shape[0], len(df), round(X_train.shape[0] / len(df) * 100)))

    print("- Test set has \033[92m{}\033[0m"
          " records out of \033[92m{}\033[0m"
          " which is \033[92m{}%\033[0m\n".format(X_test.shape[0], len(df), round(X_test.shape[0] / len(df) * 100)))

    # models and classification
    clf_models = get_classification_model()
    model_evaluation(models=clf_models,
                     X_train=X_train,
                     y_train=y_train,
                     X_test=X_test,
                     y_test=y_test,
                     show_confusion_matrix=True,
                     show_roc_curve=True,
                     show_compare_prediction_by_genre=True,
                     show_simple_compare=True)


# # used for testing
# if __name__ == '__main__':
#     classification_and_evaluation(data_path=const.DATA_PATH)
