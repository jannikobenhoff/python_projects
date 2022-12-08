import os
import gzip
import matplotlib.pyplot as plt
import numpy as np
import struct
import pandas as pd
import idx2numpy
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from joblib import dump, load
from matplotlib.ticker import MaxNLocator


def load_mnist(folder="__files", train=True):
    list = pd.DataFrame(os.listdir(folder))
    list = list[list != ".DS_Store"].dropna()
    list = list[0].values
    labels = []
    images = []
    for file in list:
        file = str(file)
        filename = folder + "/" + file
        print(filename)
        imagearray = idx2numpy.convert_from_file(filename)
        if "images" in file:
            images.append(imagearray)
        else:
            labels.append(imagearray)

    return labels, images


if __name__ == "__main__":
    labels, images = load_mnist()
    fig, axes = plt.subplots(3, 3, figsize=(7, 7))
    k = 0
    for i in range(3):
        for ii in range(3):
            axes[i, ii].imshow(images[0][k], cmap=plt.cm.binary)
            axes[i, ii].set_title(labels[0][k])
            k = k + 1
    plt.savefig("numbers.pdf")
    size = 1000

    labels = labels[0][0:size]
    images = images[0][0:size]

    x = np.array(images)
    y = np.array(labels)

    x_new = x.reshape(len(x), -1)

    xtrain, xtest, ytrain, ytest = train_test_split(x_new, y, test_size=.2, random_state=10)

    xtrain = xtrain / 255
    xtest = xtest / 255

    pca = PCA(.98)
    xtrain = pca.fit_transform(xtrain)
    xtest = pca.fit_transform(xtest)

    nn_preds = []
    nni = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    for nn in nni:
        nn = int(nn)
        knc = KNeighborsClassifier(n_neighbors=nn, algorithm="ball_tree")
        knc.fit(xtrain, ytrain)

        pred = knc.predict(xtrain)

        print("Training score:", accuracy_score(ytrain, pred))
        nn_preds.append(accuracy_score(ytrain, pred))

    fig2, ax = plt.subplots(1)
    ax = plt.figure().gca()

    ax.plot(nni, nn_preds)
    ax.set(xlabel="number of NN", ylabel="test accuracy")
    plt.xticks(np.arange(1, 21, 1))
    #ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    dump(knc, "model.sk")

    plt.close(fig)
    plt.show()
    plt.savefig("knn.pdf")
