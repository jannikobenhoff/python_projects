import pickle
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR

from helper import getWeatherData
from solarplus.lastprofil import scaleLastprofil
from solarplus.model import Model, RBF, Model2

if __name__ == "__main__":
    '''Inputs: Temperatur, Strahlung, Uhrzeit, Wochentag, So/Sa/Woche'''
    '''Outputs: Stromverbrauch, (Warmwasser, Heizverbrauch)'''
    new_input = False
    if new_input:
        ls = pd.DataFrame(scaleLastprofil(4500, 2500, 2000), columns=["index", "time", "x", "verbrauch", "art", "tag", "ww", "heiz"])
        tmy = pd.DataFrame(getWeatherData())
        df = pd.DataFrame(columns=["Temp", "Irradiance", "Time", "Monat", "Wochentag", "TagArt", "Verbrauch", "WW", "Heiz"])

        print(ls.head())
        print(tmy.head())

        df["Monat"] = tmy["time(UTC)"]
        df["Irradiance"] = tmy["G(h)"]
        df["Monat"] = df["Monat"].str[4:6]
        df["Monat"] = pd.to_numeric(df['Monat'], errors='ignore')
        df["Time"] = tmy["time(UTC)"]
        df["Time"] = df["Time"].str[9:11]
        df["Time"] = pd.to_numeric(df['Time'], errors='ignore')
        df["Wochentag"] = ls["tag"].values[0::4]
        df["TagArt"] = ls["art"].values[0::4]
        df["Verbrauch"] = ls["verbrauch"].values[0::4] * 4
        df["Heiz"] = ls["heiz"].values[0::4]
        df["WW"] = ls["ww"].values[0::4]
        df["Temp"] = tmy["T2m"]

        print(df.head())

        df.to_csv("__data/input.csv")

    df = pd.read_csv("__data/input.csv")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('A {} device was detected.'.format(device))

    # verbrauch durschnitt und standart abweichung
    # mean = df['Verbrauch'].mean()
    # std = df['Verbrauch'].std()
    # df['Verbrauch'] = (df['Verbrauch'] - mean) / std

    scaler = MinMaxScaler()

    inputs = ['Temp', 'Time', 'TagArt', 'Irradiance', 'Monat']

    x = torch.tensor(df[inputs].values, dtype=torch.float, device=device)
    scaler.fit(x)
    x = scaler.transform(x)
    x = torch.Tensor(x)

    # Extract the outputs and create a PyTorch tensor y (outputs)
    outputs = ['Verbrauch', "WW", "Heiz"]
    y = torch.tensor(df[outputs].values, dtype=torch.float, device=device)
    # shuffle tensors
    perm = torch.randperm(len(x))
    x = x[perm]
    y = y[perm]
    xtrain = x[0:6000]
    xtest = x[6000:8760]
    ytrain = y[0:6000]
    ytest = y[6000:8760]

    print(xtrain[0:3])

    model = Model(inputlayers=5, outputlayers=3)

    # Measure our neural network by mean square error
    criterion = torch.nn.MSELoss()

    # Train our network with a simple SGD optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    scheduler = ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(30):
        totalLoss = 0
        pred = [[], [], []]
        for i in range(len(xtrain)):
            # Single Forward Pass
            ypred = model.forward(xtrain[i])

            # Measure how well the model predicted vs the actual value
            loss = criterion(ypred, ytrain[i])

            # Track how well the model predicted (called loss)
            totalLoss += loss.item()

            # Update the neural network
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        # Print out our loss after each training iteration
        print("Total Loss: ", totalLoss, "Epoch: ", epoch)

    testLoss = 0
    with torch.no_grad():
        for i in range(len(xtest)):
            ypred = model.forward(xtest[i])
            loss = criterion(ypred, ytrain[i])
            testLoss += loss.item()

    print("Test Loss: ", testLoss)

    '''Safe Model'''
    with open("model.pickle", "wb") as fp:
        pickle.dump(model.state_dict(), fp)

    x = torch.tensor(df[inputs].values, dtype=torch.float)
    scaler.fit(x)
    x = scaler.transform(x)
    x = torch.Tensor(x)

    pred = [[], [], []]
    totalLoss = 0
    # model.load_state_dict(torch.load("model.pickle"))
    model.eval()
    for i in range(len(x)):
        ypred = model.forward(x[i])
        pred[0].append(ypred.detach().numpy()[0])
        pred[1].append(ypred.detach().numpy()[1])
        pred[2].append(ypred.detach().numpy()[2])
        loss = criterion(ypred, y[i])

        totalLoss += loss.item()

    print("Total Loss: ", totalLoss)

    df["VerbrauchPred"] = pred[0]
    df["WWPred"] = pred[1]
    df["HeizPred"] = pred[2]

    print(sum(df["Verbrauch"].values))
    print(sum(df["WW"].values))
    print(sum(df["Heiz"].values))
    print("---")
    print(sum(df["VerbrauchPred"].values))
    print(sum(df["WWPred"].values))
    print(sum(df["HeizPred"].values))

    window = 8500
    window2 = 0
    fig, axes = plt.subplots(1, 3, figsize=(10, 6))
    axes[0].plot(df["Verbrauch"].values[window2:window2+window], label="Verbrauch")
    axes[0].plot(df["VerbrauchPred"].values[window2:window2+window], label="Prediction")
    axes[0].legend(fontsize=8)
    axes[1].plot(df["WW"].values[window2:window2+window], label="Verbrauch")
    axes[1].plot(df["WWPred"].values[window2:window2+window], label="Prediction")
    axes[1].legend(fontsize=8)
    axes[2].plot(df["Heiz"].values[window2:window2+window], label="Verbrauch")
    axes[2].plot(df["HeizPred"].values[window2:window2+window], label="Prediction")
    axes[2].legend(fontsize=8)

    plt.show()
