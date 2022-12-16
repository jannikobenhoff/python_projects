import gzip
import struct
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, classification_report


def training_images(train=True):
    if train:
        file = '__files/train-images-idx3-ubyte.gz'
    else:
        file = '__files/t10k-images-idx3-ubyte.gz'

    with gzip.open(file, 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of images
        image_count = int.from_bytes(f.read(4), 'big')
        # third 4 bytes is the row count
        row_count = int.from_bytes(f.read(4), 'big')
        # fourth 4 bytes is the column count
        column_count = int.from_bytes(f.read(4), 'big')
        # rest is the image pixel data, each pixel is stored as an unsigned byte
        # pixel values are 0 to 255
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)\
            .reshape((image_count, row_count, column_count))
        images = images.reshape(len(images), -1)
        return images


def training_labels(train=True):
    if train:
        file = '__files/train-labels-idx1-ubyte.gz'
    else:
        file = '__files/t10k-labels-idx1-ubyte.gz'
    with gzip.open(file, 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of labels
        label_count = int.from_bytes(f.read(4), 'big')
        # rest is the label data, each label is stored as unsigned byte
        # label values are 0 to 9
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        return labels

if __name__ == "__main__":
    images_train = training_images(True)
    images_test = training_images(False)

    labels_train = training_labels(True)
    labels_test = training_labels(False)

    print(images_test.shape)
    print(images_train.shape)
    size = 1000

    labels_train = labels_train[0:size * 2]
    images_train = images_train[0:size * 2]

    labels_test = labels_test[0:size]
    images_test = images_test[0:size]

    param_C = 5
    param_gamma = 0.05*255
    svm = SVC(C=param_C) #, gamma=param_gamma)

    #svm = SVC(kernel="rbf", C=5, gamma=0.05) #LinearSVC(dual=False)
    svm.fit(images_train, labels_train)
    pred = svm.predict(images_test)
    print(accuracy_score(labels_test, pred))

    fig, axes = plt.subplots(4, 5, figsize=(10, 8))
    index = 0
    for i in range(4):
        for ii in range(5):
            im = images_test[index].reshape(28, 28)
            axes[i][ii].imshow(im)
            axes[i][ii].title.set_text("Pred: {}, True: {}".format(pred[index], labels_test[index]))
            axes[i][ii].get_xaxis().set_visible(False)
            axes[i][ii].get_yaxis().set_visible(False)
            index = index + 1

    plt.tight_layout()
    plt.savefig("__output/mnist_plot.pdf")


