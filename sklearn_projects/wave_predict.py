import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import gaussian_process
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn import preprocessing
import json
from joblib import dump, load


if __name__ == '__main__':
    df = pd.read_csv(("__files/wave.csv"))
    x = df["x"].values
    y = df["y"].values
    # lab = preprocessing.LabelEncoder()
    # y = lab.fit_transform(y)

    x = x.reshape(len(x), -1)
    # x = np.sort(x)

    scores = {"test_mae": [], "test_mse": [], "test_r2": [],
              "train_mae": [], "train_mse": [], "train_r2": []}

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.8)

    train_df = pd.DataFrame(columns=["x", "y"])
    train_df["y"] = ytrain
    train_df["x"] = xtrain
    #train_df.to_csv("train.csv", index=False, sep=",", header=True)

    test_df = pd.DataFrame(columns=["x", "y"])
    test_df["y"] = ytest
    test_df["x"] = xtest
    #test_df.to_csv("test.csv", index=False, sep=",", header=True)

    GBC = KernelRidge()
    tuned_parameters = [{"kernel": ["rbf"], "alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)}]

    hypermodel = GridSearchCV(GBC, tuned_parameters, cv=5, scoring="neg_mean_squared_error")

    hypermodel.fit(xtrain, ytrain)
    print(hypermodel.best_params_)

    x_f = np.linspace(-10, 10, 1000)[:, None]
    y_f = np.exp(-(x_f/4)**2)*np.cos(4*x_f)

    ypred = hypermodel.predict(x_f)
    r2 = r2_score(ypred, y_f)
    mse = mean_squared_error(ypred, y_f)
    mae = mean_absolute_error(ypred, y_f)

    ypred_train = hypermodel.predict(xtrain)
    scores["train_mae"] = mean_absolute_error(ypred_train, ytrain)
    scores["train_mse"] = mean_squared_error(ypred_train, ytrain)
    scores["train_r2"] = r2_score(ypred_train, ytrain)

    ypred_test = hypermodel.predict(xtest)
    scores["test_mae"] = mean_absolute_error(ypred_test, ytest)
    scores["test_mse"] = mean_squared_error(ypred_test, ytest)
    scores["test_r2"] = r2_score(ypred_test, ytest)

    with open('__output/scores.json', 'w') as f:
        json.dump(scores, f, default=str)

    print(scores)

    plt.scatter(xtrain, ytrain, label="training data", s=20)
    plt.scatter(xtest, ytest, label="test data", s=20)

    plt.plot(x_f, y_f, label="f")
    plt.plot(x_f, ypred, label="predicted f")

    plt.title("MSE: {:.3f}, MAE: {:.3f}, R\N{SUPERSCRIPT TWO}: {:.3f}".format(mse, mae, r2), fontsize=9)
    plt.legend()
    # plt.show()
    plt.savefig("__output/plot.pdf")
    dump(hypermodel, "__output/model.joblib")


