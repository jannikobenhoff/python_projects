import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from joblib import dump, load


if __name__ == '__main__':
    df = pd.read_csv("__files/credit.csv")
    list = ["Rating", "Limit", "Income", "Cards"]
    y = df["Balance"].values

    dummies = pd.get_dummies(df[list])

    X_train, X_test, y_train, y_test = train_test_split(dummies.values, y, test_size=0.20)

    r = []
    coeffs = []

    alphas = np.linspace(0, 10000, 10001)

    # l = load("model_3.joblib")
    # lp= l.predict(X_test)
    # r2 = r2_score(lp, y_test)
    # print("l r2: ", r2)

    for a in alphas:

        las = Lasso(alpha=a, max_iter=10000)
        las.fit(X_train, y_train)

        coeffs.append(las.coef_)

        y_pred = las.predict(X_test)
        r2 = r2_score(y_pred, y_test)
        r.append(r2)
        if a == 100:
            model1 = las
        elif a == 500:
            model2 = las
        elif a == 1000:
            model3 = las

    fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)

    for co in range(len(coeffs[0])):
        ax[0].plot([i[co] for i in coeffs], label=list[co])

    ax[0].set(ylabel="Value of the coefficient")
    ax[0].legend()
    ax[0].set_xscale("log")

    ax[1].set(ylabel="R\N{SUPERSCRIPT TWO}")
    ax[1].plot(alphas, r)

    plt.xlabel("alpha")
    plt.xscale("log")

    dump(model1, 'model_1.joblib')
    dump(model2, 'model_2.joblib')
    dump(model3, 'model_3.joblib')

    # plt.show()
    plt.savefig("plot.pdf")

