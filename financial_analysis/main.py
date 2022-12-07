import pandas as pd
import matplotlib.pyplot as plt
from get_ticker_data import *
import matplotlib.dates as mdates


if __name__ == "__main__":
    ticker = "hubs"

    data = pd.DataFrame(columns=["open", "date"])
    data["open"], data["date"] = get_ticker_value(ticker, "5y", "1d")

    info = get_ticker_infos(ticker)
    print("Rule of 40: {:.2f}".format(rule_of_fourty(info)))

    max_open, max_date = get_ticker_max(ticker, "5y", "1d")

    x = [str(d)[0:10] for d in data["date"]]

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax.plot(x, data["open"].values)

    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylabel("stock price USD", fontsize=10)
    ax.set_xlabel("date", fontsize=10)


    ax.annotate("Maximum n={:.2f}".format(max_open), xy=(max_date, max_open),
                xytext=(max_date, max_open*1.1),
                arrowprops=dict(facecolor='yellow', shrink=0.1, headwidth=8, headlength=8), horizontalalignment='left',
                verticalalignment='top', fontsize=7)
    ax.text(10, max_open*0.8, "Rule of 40: {:.1f}".format(rule_of_fourty(info)))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=80))
    plt.xticks(fontsize=8)
    plt.gcf().autofmt_xdate()
    plt.title("{} Stock".format(ticker))

    plt.show()