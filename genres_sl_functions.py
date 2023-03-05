import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# my import functions
import constants as const
import plot_function


def load_data(data_path, type_of_normalization):
    # read df and drop unnecessary column
    raw_dataset = pd.read_csv(data_path)
    print("\nRaw Dataset Keys:\n\033[92m{}\033[0m".format(raw_dataset.keys()))
    df = raw_dataset.drop(["filename"], axis=1)
    print("\nData Head:\n\033[92m{}\033[0m".format(df.head()))
    print("\nData Shape:\n\033[92m{}\033[0m".format(df.shape))

    # encode genre label as integer values
    # i.e.: blues = 0, ..., rock = 9
    encoder = preprocessing.OrdinalEncoder()
    df["genre"] = encoder.fit_transform(df[["genre"]])

    # split df into x and y
    label_column = "genre"
    X = df.loc[:, df.columns != label_column]
    y = df.loc[:, label_column]

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

    return X, y, df


def prepare_datasets(X, y, test_size):
    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


def get_classification_model():
    # models
    models = {"NN": [], "RF": [], "SVM": []}

    # Neural Network
    nn_model = MLPClassifier(solver="adam", alpha=1e-5, hidden_layer_sizes=(16, 16), random_state=1,
                             activation="relu", learning_rate="adaptive", early_stopping=True, verbose=False,
                             max_iter=200)
    models.update({"NN": nn_model})

    # Random forest
    rf_model = RandomForestClassifier(n_estimators=10, random_state=10)
    models.update({"RF": rf_model})

    # Support Vector Machine
    svc_model = SVC(C=10, kernel="rbf", probability=True, random_state=10)
    models.update({"SVM": svc_model})

    return models


def model_evaluation(models, X_train, y_train, X_test, y_test, show_confusion_matrix=True, show_roc_curve=True):
    # evaluation of every classification model
    for key, value in models.items():
        model_name = key
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
                                   genres_list=const.GENRES_SET,
                                   type_of_learning="SL",
                                   show_on_screen=True,
                                   store_in_folder=True)

    # per ogni clf_model
    # esegue il plot di:
    # confusion matrix, F
    # roc curve F
    # prediction
    # definire file cls con accuracy


def classification_processes_and_evaluation(data_path, normalization_type):
    # load data
    X, y, df = load_data(data_path=data_path, type_of_normalization=normalization_type)
    print("\nData:\n\033[92m{}\033[0m".format(df))
    print("\nX (my data):\n\033[92m{}\033[0m".format(X))
    print("\ny (labels):\n\033[92m{}\033[0m".format(y))

    # create train/test split
    X_train, X_test, y_train, y_test = prepare_datasets(X=X, y=y, test_size=0.3)
    print("\nSplit data into Train and Test:")
    print("- Train set has \033[92m{}\033[0m"
          " records out of \033[92m{}\033[0m"
          " which is {}%".format(X_train.shape[0], len(df), round(X_train.shape[0] / len(df) * 100)))

    print("- Test set has \033[92m{}\033[0m"
          " records out of \033[92m{}\033[0m"
          " which is \033[92m{}%\033[0m".format(X_test.shape[0], len(df), round(X_test.shape[0] / len(df) * 100)))

    print("\nStarting Classification Proces: ")
    clf_models = get_classification_model()
    model_evaluation(models=clf_models, X_train=X_train, y_train=y_train, X_test=X_test,
                     y_test=y_test, show_confusion_matrix=True, show_roc_curve=True)


if __name__ == '__main__':
    classification_processes_and_evaluation(const.DATA_PATH, "min_max")
