import math

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
import pickle

from solarplus.lastprofil import scaleLastprofil
from solarplus.model import Model
from solarplus.pv_ertrag import pv_performance
import tensorflow as tf
from tensorflow import keras
from keras import layers

def getWeatherData():
    """
    time: time(UTC)
    temperature: T2m
    relative humidity: RH
    global horizontal irradiance: G(h)
    direct irradiance: Gb(n)
    diffuse horizontal irradiance: Gd(n)
    infrared radiation downwards: IR(h)
    wind speed: WS10m
    wind direction: WD10m
    surface air pressure: SP
    """
    df = pd.read_json("__data/tmy.json")
    df = df["outputs"]["tmy_hourly"]
    return df


def getApiResponse(strom, ww, heiz, pv, dach):
    ls = pd.DataFrame(scaleLastprofil(strom, ww, heiz),
                      columns=["index", "time", "x", "verbrauch", "art", "tag", "ww", "heiz"])
    tmy = pd.DataFrame(getWeatherData())
    df = pd.DataFrame(columns=
                      ["Date", "Temp", "Irradiance", "Time", "Monat",
                       "Wochentag", "TagArt", "Verbrauch", "WW", "Heiz", "Ertrag", ]
                       #"Facts"]
                      )

    np.seterr(divide='ignore')
    '''Load Model'''
    model = Model(inputlayers=5, outputlayers=3)
    with open("model.pickle", "rb") as fp:
        model.load_state_dict(pickle.load(fp))


    df["Temp"] = tmy["T2m"]
    df["Monat"] = tmy["time(UTC)"]
    df["Irradiance"] = tmy["G(h)"]
    df["Monat"] = df["Monat"].str[4:6]
    df["Monat"] = pd.to_numeric(df['Monat'], errors='ignore')
    df["Time"] = tmy["time(UTC)"]
    df["Date"] = tmy["time(UTC)"]
    df["Time"] = df["Time"].str[9:11]
    df["Time"] = pd.to_numeric(df['Time'], errors='ignore')
    df["Wochentag"] = ls["tag"].values[0::4]
    df["TagArt"] = ls["art"].values[0::4]

    inputs = ['Temp', 'Time', 'TagArt', 'Irradiance', 'Monat']

    '''Load Tensorflow Model'''
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(df[inputs]))
    model = tf.keras.Sequential([
        normalizer,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='sigmoid'),
    ])
    latest = tf.train.latest_checkpoint("training")

    model.load_weights(latest)

    df[["Verbrauch", "WW", "Heiz"]] = model.predict(df[inputs])
    #
    # x = torch.tensor(df[inputs].values, dtype=torch.float)
    # scaler = MinMaxScaler()
    #
    # scaler.fit(x)
    # x = scaler.transform(x)
    # x = torch.Tensor(x)
    # print(x[0:3])
    # pred = [[], [], []]
    # # model.eval()
    # for i in range(len(x)):
    #     ypred = model(x[i])
    #     pred[0].append(ypred.detach().numpy()[0])
    #     pred[1].append(ypred.detach().numpy()[1])
    #     pred[2].append(ypred.detach().numpy()[2])
    #
    # df["Verbrauch"] = pred[0]
    # df["WW"] = pred[1]
    # df["Heiz"] = pred[2]
    df["Ertrag"] = pv_performance(pv, tmy["G(h)"], tmy["T2m"], tmy["WD10m"])
    df["Ertrag"] = df["Ertrag"].fillna(0) * int(dach/2)

    df["Verbrauch"] = 1000 * df["Verbrauch"] * (strom / sum(df["Verbrauch"].values))
    df["Heiz"] = 1000 * df["Heiz"] * (heiz/ sum(df["Heiz"].values))
    df["WW"] = 1000 * df["WW"] * (ww/sum(df["WW"].values))
    print(sum(df["Ertrag"].values), sum(df["Verbrauch"].values),
                                      sum(df["WW"].values),
                                      sum(df["Heiz"].values))
    j = df.to_dict("list")
    j["Facts"] = {"gesamtErtrag": sum(j["Ertrag"]),
                  "gesamtVerbrauch": sum(j["WW"]) + sum(j["Heiz"]) + sum(j["Verbrauch"]),
                  "gedeckteStunden": sum(list(map(int, df["Ertrag"] > (df["WW"]+df["Heiz"]+df["Verbrauch"]))))}
    return j


if __name__ == "__main__":
    df = getApiResponse(4000, 2500, 2000, "Si", 100)
    fig, axes = plt.subplots(1, 3)
    axes[0].plot(df["Verbrauch"])
    axes[1].plot(df["WW"])
    axes[2].plot(df["Heiz"])

    plt.show()
