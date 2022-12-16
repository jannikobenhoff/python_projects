import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def find_hyperparams(base_model, paramgrid, features, targets, cv=5, **kwopts):
    model = GridSearchCV(base_model, paramgrid, cv=cv, n_jobs=-1) #, scoring="neg_mean_squared_error")
    model.fit(features, targets)
    return model


def plot_decision_boundary(clf, X, Y, cmap='plasma'):
    h = 0.02
    x_min, x_max = 3.5, 8.5 #X[:,0].min() - 10*h, X[:,0].max() + 10*h
    y_min, y_max = 1, 5.5 #X[:,1].min() - 10*h, X[:,1].max() + 10*h
    print(x_min, x_max, y_min, y_max)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.25)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.7)
    plt.scatter(X[:,0], X[:,1], c=Y, cmap=cmap, edgecolors='k');
    plt.axis([3.5, 8.5, 1, 5.5])


if __name__ == "__main__":
    iris = load_iris()
    n_neighbors = 15
    cv = 5

    x = iris.data[:, :2]
    y = iris.target
    names = iris.target_names
    print(names)

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3)

    test_farbe = []
    train_farbe = []
    train_names = []
    colors = ["blue", "red", "yellow"]
    for i in range(len(ytest)):
        if ytest[i] == 0:
            test_farbe.append("blue")
        elif ytest[i] == 1:
            test_farbe.append("red")
        else:
            test_farbe.append("yellow")
    for i in range(len(ytrain)):
        if ytrain[i] == 0:
            train_farbe.append("blue")
            train_names.append("Iris-"+names[0])
        elif ytrain[i] == 1:
            train_farbe.append("red")
            train_names.append("Iris-"+names[1])
        else:
            train_farbe.append("yellow")
            train_names.append("Iris-"+names[2])

    poly_parameters = [{'kernel': ['poly'], "C": np.linspace(1, 10, 5),
                        "degree": [2, 3, 4], "coef0": [0.0, 0.5, 1]}]

    rbf_parameters = [{"kernel": ["rbf"], "C": np.linspace(0.001, 1, 10)}]

    linear_parameters = [{"kernel": ["linear"], "C": np.linspace(0.001, 10, 10)}]

    grid_params = {'n_neighbors': [5, 7, 9, 11, 13, 25, 33, 35, 38],
                   'weights': ['uniform', 'distance'],
                   'metric': ['minkowski', 'euclidean', 'manhattan']}

    knn = find_hyperparams(KNeighborsClassifier(), grid_params, x, y)
    svm_linear = find_hyperparams(SVC(), linear_parameters, x, y)
    svm_poly = find_hyperparams(SVC(), poly_parameters, x, y)
    svm_rbf = find_hyperparams(SVC(), rbf_parameters, x, y)

    models = [[knn, svm_linear], [svm_poly, svm_rbf]]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    for i in range(2):
        for ii in range(2):
            model = models[i][ii]
            ypredict = model.predict(xtest)
            testscore = accuracy_score(ypredict, ytest)
            trainscore = model.best_score_
            params = model.best_params_
            x_min, x_max = 3.5, 8.5
            y_min, y_max = 1, 5.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                                 np.arange(y_min, y_max, 0.01))
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            cs = axes[i][ii].contourf(xx, yy, Z, cmap="plasma", alpha=0.75)
            axes[i][ii].contour(xx, yy, Z, colors='k', linewidths=0.7)
            for spec in [0, 1, 2]:
                idx_test = np.where(ytest == spec)
                idx_train = np.where(ytrain == spec)
                axes[i][ii].scatter(xtest[:, 0][idx_test], xtest[:, 1][idx_test], color=colors[spec], marker="+")
                axes[i][ii].scatter(xtrain[:, 0][idx_train], xtrain[:, 1][idx_train], s=15, c=colors[spec],
                                    edgecolors='k', label="Iris-"+names[spec])
            axes[i][ii].axis([x_min, x_max, y_min, y_max])
            axes[i][ii].set_title("{}, train: {:.2f}, test: {:.2f}\n"
                                  "{}".format(str(model.estimator)[0:-2], trainscore, testscore, params),
                                  fontsize=8)
            axes[i][ii].legend(fontsize=8)
    plt.tight_layout()
    # plt.show()
    plt.savefig("__output/iris_plot.pdf")
