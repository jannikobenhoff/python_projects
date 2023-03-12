import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras import layers
from solarplus.helper import getWeatherData
from solarplus.lastprofil import scaleLastprofil
import seaborn as sns

from solarplus.model import Model

if __name__ == "__main__":
    '''Inputs: Temperatur, Strahlung, Uhrzeit, Tagart'''
    '''Outputs: Stromverbrauch, (Warmwasser, Heizverbrauch)'''
    new_input = False
    if new_input:
        ls = pd.DataFrame(scaleLastprofil(4500, 2500, 2000),
                          columns=["index", "time", "x", "strom", "art", "wochentag", "ww", "heiz"])
        tmy = pd.DataFrame(getWeatherData())
        df = pd.DataFrame(
            columns=["temp", "strahlung", "hour", "month", "art", "strom", "ww", "heiz"])

        df["month"] = tmy["time(UTC)"]
        df["strahlung"] = tmy["G(h)"]
        df["month"] = df["month"].str[4:6]
        df["month"] = pd.to_numeric(df['month'], errors='ignore')
        df["hour"] = tmy["time(UTC)"]
        df["hour"] = df["hour"].str[9:11]
        df["hour"] = pd.to_numeric(df['hour'], errors='ignore')
        df["art"] = ls["art"].values[0::4]
        df["strom"] = ls["strom"].values[0::4] * 4
        df["heiz"] = ls["heiz"].values[0::4]
        df["ww"] = ls["ww"].values[0::4]
        df["temp"] = tmy["T2m"]

        print(df.head())

        df.to_csv("__data/input.csv")

    df = pd.read_csv("__data/input.csv")

    inputs = ['temp', 'hour', 'art', 'strahlung', 'month', 'strom', 'heiz', 'ww']
    outputs = ['strom', "ww", "heiz"]
    df = df[inputs]
    train_dataset = df.sample(frac=0.8, random_state=0)
    test_dataset = df.drop(train_dataset.index)

    # sns.pairplot(train_dataset, diag_kind='kde')
    # plt.show()

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = pd.DataFrame([train_features.pop(x) for x in outputs]).transpose()
    test_labels = pd.DataFrame([test_features.pop(x) for x in outputs]).transpose()

    print(max(train_labels["strom"].values))

    scaler = MinMaxScaler()

    scaler.fit(train_labels["strom"].values.reshape(-1, 1))
    train_labels["strom"] = scaler.transform(train_labels["strom"].values.reshape(-1, 1))

    print(max(train_labels["strom"].values))

    print(train_labels.head())
    print(train_features.head())

    print(train_labels.shape)
    print(train_features.shape)

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))

    checkpoint_path = "training/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    linear_model = tf.keras.Sequential([
        normalizer,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='sigmoid'),
    ])

    # linear_model.summary()

    linear_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
        loss='mean_absolute_error')

    history = linear_model.fit(
        train_features,
        train_labels,
        epochs=200,
        # Suppress logging.
        # verbose=0,
        # Calculate validation results on 20% of the training data.
        validation_split=0.2,
        callbacks=[cp_callback]
    )

    df = pd.read_csv("__data/input.csv")
    inputs = ['temp', 'hour', 'art', 'strahlung', 'month']

    pred = linear_model.predict(df[inputs])
    df1 = pd.DataFrame(pred, columns=["VerbrauchPred", "WWPred", "HeizPred"])
    print(max(df1["VerbrauchPred"].values))
    df1["VerbrauchPred"] = scaler.inverse_transform(df1["VerbrauchPred"].values.reshape(-1, 1))
    print(max(df1["VerbrauchPred"].values))

    '''Load Model'''
    model = Model(inputlayers=5, outputlayers=3)
    with open("model.pickle", "rb") as fp:
        model.load_state_dict(pickle.load(fp))

    x = torch.tensor(df[inputs].values, dtype=torch.float)
    scaler = MinMaxScaler()

    scaler.fit(x)
    x = scaler.transform(x)
    x = torch.Tensor(x)

    pred2 = [[], [], []]
    # model.eval()
    for i in range(len(x)):
        ypred = model(x[i])
        pred2[0].append(ypred.detach().numpy()[0])
        pred2[1].append(ypred.detach().numpy()[1])
        pred2[2].append(ypred.detach().numpy()[2])

    df2 = pd.DataFrame(columns=["VerbrauchPred", "WWPred", "HeizPred"])
    df2["VerbrauchPred"] = pred2[0]
    df2["WWPred"] = pred2[1]
    df2["HeizPred"] = pred2[2]

    fig, axes = plt.subplots(1, 4)
    axes[0].plot(df1.values - df[outputs].values)
    axes[1].plot(df2.values - df[outputs].values)
    axes[2].plot(df[outputs])
    axes[3].plot(df1)
    plt.show()


    def plot_loss(history):
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        # plt.ylim([0, 10])
        plt.xlabel('Epoch')
        plt.ylabel('Error [MPG]')
        plt.legend()
        plt.grid(True)
        plt.show()


    plot_loss(history)
