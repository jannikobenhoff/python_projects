import csv
import math

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
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


def getPvgisData(neigung, azimut, peak, pv):
    vgl = getVglJahr()
    vglString = ('200601', '201002', '200503', '201604', '200905', '201406', '200907', '201508', '200809', '201510', '201111', '201412')

    url = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
    params = {
        "lat": 48.271,
        "lon": 11.581,
        "outputformat": "json",
        "startyear": min(vgl),
        "endyear": max(vgl),
        "pvtechchoice": pv,
        "peakpower": peak,
        "pvcalculation": 1,
        "mountingplace": "building",
        "loss": 15,
        "angle": neigung,
        "aspect": azimut,

    }

    data = requests.get(url, params)

    df = pd.DataFrame(data.json()["outputs"]["hourly"])
    df = df.rename(columns={"T2m": "temp", "G(i)": "strahlung"})
    print(df.shape)
    df = df[df.time.str.startswith(vglString)]
    # df["year"] = df["time"].str[0:4]
    # df["year"] = pd.to_numeric(df['year'], errors='ignore')

    df["month"] = df["time"].str[4:6]
    df["month"] = pd.to_numeric(df['month'], errors='ignore')

    df["day"] = df["time"].str[6:8]
    df["day"] = pd.to_numeric(df['day'], errors='ignore')

    df["hour"] = df["time"].str[9:11]
    df["hour"] = pd.to_numeric(df['hour'], errors='ignore')

    df = df.drop(columns=["time"])
    df = df.sort_values(by=["month", "day", "hour"])

    return df

def getVglJahr():
    # vergleichsjahr = [{"month": 1, "year": 2016}, {"month": 2, "year": 2016}, {"month": 3, "year": 2020},
    #                   {"month": 4, "year": 2008}, {"month": 5, "year": 2006}, {"month": 6, "year": 2010},
    #                   {"month": 7, "year": 2014}, {"month": 8, "year": 2014}, {"month": 9, "year": 2015},
    #                   {"month": 10, "year": 2017}, {"month": 11, "year": 2011}, {"month": 12, "year": 2018}]
    vergleichsjahr = [2006,
                      2016,
                      2005,
                      2016,
                      2009,
                      2014,
                      2009,
                      2015,
                      2008,
                      2015,
                      2011,
                      2014]
    return vergleichsjahr
    lat = 48.271
    long = 11.581
    url = "https://re.jrc.ec.europa.eu/api/tmy?lat={}&lon={}".format(lat, long)

    response = requests.get(url=url)
    decoded_content = response.content.decode('utf-8')
    cr = csv.reader(decoded_content.splitlines(), delimiter=',')
    my_list = list(cr)
    print("Vergleichsjahr: " + " - ".join(map(str, my_list[4:16])))
    return my_list[4:16]


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


def calcPeak(dach, dachart):
    """
    Dachart:
        0 - Satteldach
        1 - Pultdach
        2 - Flachdach
        3 - Walmdach
    PV:
        "crystSiMono" - Mono Crystalline Silicon - 345 Wp pro Modul -0,2 kWp pro m2
        "crystSiPoly" - Poly Crystalline Silicon - 280 Wp pro Modul
        "CIS" - CIS - 4,3 kWp pro 50m2
        "CdTe" - Cadmiumtellurid - 3,5 kWp pro 50m2
    """


def getApiResponse(strom, ww, heiz, pv, azimut, neigung, dachart, dach):
    ls = pd.DataFrame(scaleLastprofil(strom, ww, heiz),
                      columns=["index", "time", "x", "verbrauch", "art", "tag", "ww", "heiz"])
    #tmy = pd.DataFrame(getWeatherData())
    # df = pd.DataFrame(columns=
    #                   ["Date", "Temp", "Irradiance", "Time", "Monat",
    #                    "Wochentag", "TagArt", "Verbrauch", "WW", "Heiz", "Ertrag", ]
    #                   # "Facts"]
    #                   )
    peak = calcPeak(dach=dach, dachart=dachart)
    df = getPvgisData(azimut=azimut, neigung=neigung, peak=peak, pv=pv)
    df["art"] = ls["art"].values[0::4]
    print(df.head())
    print(df.columns.to_list())
    np.seterr(divide='ignore')
    '''Load Model'''
    model = Model(inputlayers=5, outputlayers=3)
    with open("model.pickle", "rb") as fp:
        model.load_state_dict(pickle.load(fp))

    # df["Temp"] = tmy["T2m"]
    # df["Monat"] = tmy["time(UTC)"]
    # df["Irradiance"] = tmy["G(h)"]
    # df["Monat"] = df["Monat"].str[4:6]
    # df["Monat"] = pd.to_numeric(df['Monat'], errors='ignore')
    # df["Time"] = tmy["time(UTC)"]
    # df["Date"] = tmy["time(UTC)"]
    # df["Time"] = df["Time"].str[9:11]
    # df["Time"] = pd.to_numeric(df['Time'], errors='ignore')
    # df["Wochentag"] = ls["tag"].values[0::4]
    # df["TagArt"] = ls["art"].values[0::4]

    inputs = ['temp', 'hour', 'art', 'strahlung', 'month']

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
    print(df[inputs].shape)
    df[["strom", "ww", "heiz"]] = model.predict(df[inputs])
    print(df.head())
    df.to_csv("__data/t.csv")
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
    # df["Ertrag"] = pv_performance(pv, tmy["G(h)"], tmy["T2m"], tmy["WD10m"])
    # df["Ertrag"] = df["Ertrag"].fillna(0) * int(dach / 2)
    print(sum(df["P"].values), sum(df["strom"].values),
          sum(df["ww"].values),
          sum(df["heiz"].values))
    df["strom"] = 1000 * df["strom"] * (strom / sum(df["strom"].values))
    df["heiz"] = 1000 * df["heiz"] * (heiz / sum(df["heiz"].values))
    df["ww"] = 1000 * df["ww"] * (ww / sum(df["ww"].values))
    print(sum(df["P"].values), sum(df["strom"].values),
          sum(df["ww"].values),
          sum(df["heiz"].values))
    j = df.to_dict("list")
    j["Facts"] = {"gesamtErtrag": sum(j["P"]),
                  "gesamtVerbrauch": sum(j["ww"]) + sum(j["heiz"]) + sum(j["strom"]),
                  "gedeckteStunden": sum(list(map(int, df["P"] > (df["ww"] + df["heiz"] + df["strom"]))))}
    return j


if __name__ == "__main__":
    df = getApiResponse(strom=4000, ww=2500, heiz=2000, pv="crystSi", azimut=0, neigung=30, dachart=0)
    fig, axes = plt.subplots(1, 3)
    axes[0].plot(df["strom"])
    axes[1].plot(df["ww"])
    axes[2].plot(df["heiz"])

    plt.show()

