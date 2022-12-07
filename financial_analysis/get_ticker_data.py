import requests
import datetime as dt
import matplotlib.pyplot as plt
import math
import yfinance as yf
from datetime import datetime
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


def get_ticker_value(ticker, range, interval):
    base_url = 'https://query1.finance.yahoo.com'
    url = "{}/v8/finance/chart/{}?range={}&interval={}".format(base_url, ticker, range, interval)
    df = pd.read_json(url)
    close_list = df["chart"]["result"][0]["indicators"]["quote"][0]["close"]

    date_list = [datetime.fromtimestamp(x) for x in df["chart"]["result"][0]["timestamp"]]

    return close_list, date_list


def get_ticker_infos(ticker):
    headers = {
        'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36"}

    stock = ticker
    URL = 'https://finance.yahoo.com/quote/' + stock + '/key-statistics?p=' + stock
    print("url: ", URL)
    page = requests.get(URL, headers=headers)

    soup = BeautifulSoup(page.content, 'lxml')
    table_data = soup.find_all('td')

    df = pd.DataFrame(columns=['Metric Name', 'Metric'])

    for data in range(0, len(table_data) - 1, 2):
        df = pd.concat([df, pd.DataFrame({'Metric Name': table_data[data].text, 'Metric': table_data[data + 1].text}, index=[0])], ignore_index=True)

    return df


def rule_of_fourty(df: pd.DataFrame):
    print(df)
    profit_margin = df.loc[df['Metric Name'].str.contains("Profit Margin", case=False)]["Metric"].values[0]
    operating_margin = df.loc[df['Metric Name'].str.contains("Operating Margin", case=False)]["Metric"].values[0]
    value_revenue = df.loc[df['Metric Name'].str.contains("Enterprise Value/Revenue", case=False)]["Metric"].values[0]
    growth = df.loc[df['Metric Name'].str.contains("Quarterly Revenue Growth", case=False)]["Metric"].values[0]

    rule = float(growth.strip("%")) + float(operating_margin.strip("%"))

    return rule


def get_ticker_max(ticker, range, interval):
    base_url = 'https://query1.finance.yahoo.com'
    url = "{}/v8/finance/chart/{}?range={}&interval={}".format(base_url, ticker, range, interval)
    df = pd.read_json(url)
    close_list = df["chart"]["result"][0]["indicators"]["quote"][0]["close"]
    max_index = close_list.index(max(close_list))
    date_list = [datetime.fromtimestamp(x) for x in df["chart"]["result"][0]["timestamp"]]

    return close_list[max_index], str(date_list[max_index])[0:10]


