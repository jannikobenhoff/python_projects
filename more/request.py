import requests
from bs4 import BeautifulSoup
import pandas as pd


if __name__ == "__main__":
    page = requests.get("https://neuraum.de/")
    soup = BeautifulSoup(page.content, 'lxml')
    table_data = soup.find_all("div", {"class": "e-con-inner"})
    for i in range(len(table_data)):
        print(table_data[i].text.split())
    pd.Da
    page = requests.get("https://www.blitz.club/program/")
    soup = BeautifulSoup(page.content, 'lxml')
    table_data = soup.find_all("div", {"class": "short"})
    for i in range(len(table_data)):
        print(table_data[i].text.split())

    page = requests.get("https://www.muffatwerk.de/en/events/party")
    soup = BeautifulSoup(page.content, 'lxml')
    table_data = soup.find_all("div", {"class": "row"})
    for i in range(len(table_data)):
        print(table_data[i].text.split())

    # page = requests.get("https://www.ravestreamradio.de/raveevents")
    # soup = BeautifulSoup(page.content, 'lxml')
    # table_data = soup.find_all("div", {"class": "_3iVFe", "role": "listitem"})
    # for i in range(len(table_data)):
    #     print(table_data[i].text.split())
