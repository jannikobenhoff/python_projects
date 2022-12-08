import pandas as pd
import matplotlib.pyplot as plt
from get_ticker_data import *
import matplotlib.dates as mdates


def plot_sma(ticker, days):
    data = pd.DataFrame(columns=["open", "date"])
    data["open"], data["date"] = get_ticker_value(ticker, "3y", "1d")
    x = [str(d)[0:10] for d in data["date"]]

    sma_list = calc_sma(data["open"].values, days)

    fig, ax = plt.subplots(1, 1)

    ax.plot(x, data["open"].values)
    ax.plot(sma_list)
    ax.set_xlabel("Date", fontsize=8)
    ax.set_ylabel("Stock Price (USD)", fontsize=8)

    plt.title("Simple Moving Average (SMA) of {}".format(ticker))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=80))
    plt.xticks(fontsize=8)
    plt.gcf().autofmt_xdate()
    plt.show()


def calc_sma(val, days):
    sma = []
    for i in range(1, len(val), 1):
        if i < days:
            k = -i+days
            s = sum(val[i - days + k:i]) / i
        else:
            k = 0
            s = sum(val[i - days + k:i]) / days
        sma.append(s)
    return sma


def plot_ema(ticker, days):
    data = pd.DataFrame(columns=["open", "date"])
    data["open"], data["date"] = get_ticker_value(ticker, "3y", "1d")
    x = [str(d)[0:10] for d in data["date"]]

    ema_list = calc_ema(data["open"].values, days)

    fig, ax = plt.subplots(1, 1)

    ax.plot(x, data["open"].values)
    ax.plot(ema_list)
    ax.set_xlabel("Date", fontsize=8)
    ax.set_ylabel("Stock Price (USD)", fontsize=8)

    plt.title("Exponential Moving Average (EMA) of {}".format(ticker))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=80))
    plt.xticks(fontsize=8)
    plt.gcf().autofmt_xdate()
    plt.show()

def calc_ema(val, days):
    ema = [val[0]]
    multi = 2/(days+1)
    for i in range(1, len(val)):
        e = multi * val[i] + ema[i-1]*(1-multi)
        ema.append(e)
    return ema


def plot_sma_and_ema(ticker, days):
    data = pd.DataFrame(columns=["open", "date"])
    data["open"], data["date"] = get_ticker_value(ticker, "3y", "1d")
    x = [str(d)[0:10] for d in data["date"]]

    ema_list = calc_ema(data["open"].values, days)
    sma_list = calc_sma(data["open"].values, days)

    fig, ax = plt.subplots(1, 1)

    ax.plot(x, data["open"].values)
    ax.plot(ema_list, label="EMA")
    ax.plot(sma_list, label="SMA")
    ax.set_xlabel("Date", fontsize=8)
    ax.set_ylabel("Stock Price (USD)", fontsize=8)

    plt.title("Exponential Moving Average (EMA) of {}".format(ticker))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=80))
    plt.xticks(fontsize=8)
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.show()


