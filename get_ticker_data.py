import requests
import datetime as dt
import matplotlib.pyplot as plt
import math
from datetime import datetime
import numpy as np
import pandas as pd

def get_ticker_value(ticker, range, interval):
    base_url = 'https://query1.finance.yahoo.com'
    url = "{}/v8/finance/chart/{}?range={}&interval={}".format(base_url, ticker, range, interval)
    df = pd.read_json(url)
    close_list = df["chart"]["result"][0]["indicators"]["quote"][0]["close"]

    date_list = datetime.fromtimestamp(df["chart"]["result"][0]["timestamp"][0])

    return close_list, date_list


def get_ticker_infos(ticker):
    base_url = 'https://query1.finance.yahoo.com'
    url = "{}/v8/finance/chart/{}".format(base_url, ticker)
    df = pd.read_json(url)

    return df["chart"]["result"][0]["meta"]

