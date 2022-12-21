import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import gaussian_process
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score
import json
from joblib import dump, load


if __name__ == '__main__':
    df = pd.read_csv(("__files/wave.csv"))
    x = df["x"].values
    y = df["y"].values

    scores = {"test_mae": [], "test_mse": [], "test_r2": [],
              "train_mae": [], "train_mse": [], "train_r2": []}

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.8, random_state=5)

    krr = KernelRidge()
    tuned_parameters = [{"kernel": ["rbf"], "alpha": [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 10, 100,
                                                      1000],
                         'gamma': [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100]}]

    hypermodel = GridSearchCV(krr, tuned_parameters, cv=5, scoring="neg_mean_squared_error")

    hypermodel.fit(xtrain.reshape(len(xtrain), -1), ytrain)
    best = hypermodel.best_params_

    dump(hypermodel, "__output/wave_model.joblib")

    x_f = np.linspace(-10, 10, 500)[:, None]
    y_f = np.exp(-(x_f/4)**2)*np.cos(4*x_f)

    ypred = hypermodel.predict(xtest.reshape(len(xtest), -1))
    ypred_f = hypermodel.predict(x_f)
    r2 = r2_score(ypred, ytest)
    mse = mean_squared_error(ypred, ytest)
    mae = mean_absolute_error(ypred, ytest)

    ypred_train = hypermodel.predict(xtrain.reshape(len(xtrain), -1))
    scores["train_mae"] = mean_absolute_error(ypred_train, ytrain)
    scores["train_mse"] = mean_squared_error(ypred_train, ytrain)
    scores["train_r2"] = r2_score(ypred_train, ytrain)

    ypred_test = hypermodel.predict(xtest.reshape(len(xtest), -1))
    scores["test_mae"] = mean_absolute_error(ypred_test, ytest)
    scores["test_mse"] = mean_squared_error(ypred_test, ytest)
    scores["test_r2"] = r2_score(ypred_test, ytest)


    with open('__output/wave_scores.json', 'w') as f:
        json.dump(scores, f, default=str)

    print(scores)

    plt.scatter(xtrain, ytrain, label="training data", s=20)
    plt.scatter(xtest, ytest, label="test data", s=20)

    plt.plot(x_f, y_f, label="f")
    plt.plot(x_f, ypred_f, label="predicted f")

    plt.title("MSE: {:.3f}, MAE: {:.3f}, R\N{SUPERSCRIPT TWO}: {:.3f}".format(mse, mae, r2), fontsize=9)
    plt.legend()
    plt.savefig("__output/wave_plot.pdf")


