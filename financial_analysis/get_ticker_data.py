import urllib.error
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
    print("url: ", url)

    try:
        df = pd.read_json(url)

        close_list = df["chart"]["result"][0]["indicators"]["quote"][0]["close"]

        date_list = [datetime.fromtimestamp(x) for x in df["chart"]["result"][0]["timestamp"]]
        return close_list, date_list

    except urllib.error.HTTPError:
        print("http error")
        return pd.DataFrame(columns=["open", "date"])




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
    if len(table_data) == 0 or "Market Cap" not in str(table_data[0].text):
        return df

    for data in range(0, len(table_data) - 1, 2):
        df = pd.concat([df, pd.DataFrame({'Metric Name': table_data[data].text, 'Metric': table_data[data + 1].text}, index=[0])], ignore_index=True)

    return df, URL


def rule_of_fourty(df: pd.DataFrame):
    if df.empty:
        return 0, False

    profit_margin = df.loc[df['Metric Name'].str.contains("Profit Margin", case=False)]["Metric"].values[0]
    operating_margin = df.loc[df['Metric Name'].str.contains("Operating Margin", case=False)]["Metric"].values[0]
    value_revenue = df.loc[df['Metric Name'].str.contains("Enterprise Value/Revenue", case=False)]["Metric"].values[0]
    growth = df.loc[df['Metric Name'].str.contains("Quarterly Revenue Growth", case=False)]["Metric"].values[0]


    if profit_margin == "N/A" or operating_margin == "N/A" or value_revenue == "N/A" or growth == "N/A":
        return 0, False

    rule = float(growth.replace("%", "").replace(",", "")) + float(operating_margin.replace("%", "").replace(",", ""))

    if rule > 40 and 0 < float(value_revenue) <10 and float(profit_margin.strip("%"))/100 > 0 and float(operating_margin.strip("%"))/100 > 0:
        good = True
    else:
        good = False

    return rule, good, float(value_revenue), float(profit_margin.strip("%"))/100


def get_ticker_max(ticker, range, interval):
    base_url = 'https://query1.finance.yahoo.com'
    url = "{}/v8/finance/chart/{}?range={}&interval={}".format(base_url, ticker, range, interval)
    try:
        df = pd.read_json(url)
        close_list = df["chart"]["result"][0]["indicators"]["quote"][0]["close"]
        max_index = close_list.index(max(close_list))
        date_list = [datetime.fromtimestamp(x) for x in df["chart"]["result"][0]["timestamp"]]
    except urllib.error.HTTPError:
        return 0, 0

    return close_list[max_index], str(date_list[max_index])[0:10]

