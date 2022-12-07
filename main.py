import pandas as pd

from get_ticker_data import *


if __name__ == "__main__":
    data = pd.DataFrame(columns=["open", "date"])
    data["open"], data["date"] = get_ticker_value("aapl", "5y", "1d")

    info = get_ticker_infos("aapl")
    print(info)