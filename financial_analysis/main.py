import pandas as pd
import matplotlib.pyplot as plt
from get_ticker_data import *
import matplotlib.dates as mdates
import os as os


if __name__ == "__main__":
    range = "3y"

    tickers = pd.read_csv("__files/tickers_nasdaq.csv")
    tickers = tickers["Symbol"].values.tolist()

    output_list = (os.listdir("__output"))
    output_list.sort()
    if len(output_list) == 0:
        last_file_index = 0
    else:
        last_file = output_list[-1].split(".")[0]
        last_file_index = tickers.index(last_file)

    for ticker in tickers[last_file_index:-1]:
        print(ticker)
        data = pd.DataFrame(columns=["open", "date"])
        data["open"], data["date"] = get_ticker_value(ticker, range, "1d")

        if data["open"].empty:
            continue

        info, url = get_ticker_infos(ticker)

        if info.empty:
            continue

        r40, good, value_rev, prof_marg = rule_of_fourty(info)

        if good == False:
            continue

        max_open, max_date = get_ticker_max(ticker, range, "1d")

        if max_open == 0:
            continue

        x = [str(d)[0:10] for d in data["date"]]

        fig, ax = plt.subplots(1, 1, figsize=(9, 6))

        ax.plot(x, data["open"].values)

        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_ylabel("stock price USD", fontsize=10)
        ax.set_xlabel("date", fontsize=10)

        ax.annotate("Maximum = {:.2f}$".format(max_open), xy=(max_date, max_open),
                    xytext=(max_date, max_open*1.1),
                    arrowprops=dict(facecolor='yellow', shrink=0.1, headwidth=8, headlength=8), horizontalalignment='left',
                    verticalalignment='top', fontsize=7)

        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=80))
        plt.xticks(fontsize=8)
        plt.gcf().autofmt_xdate()
        plt.title("{} Stock".format(ticker))
        plt.figtext(0.04, 0.7, "Rule of 40: {:.1f}\n\n"
                                "Profit Margin: {:.2f}%\n\n"
                                "Value/Revenue: {}\n\n"
                                "Click here for all stats.".format(r40, prof_marg, value_rev),
                                fontsize=9, url=url)
        #plt.grid(True)
        plt.subplots_adjust(left=0.35)
        #plt.show()
        plt.savefig("__output/{}.pdf".format(ticker))