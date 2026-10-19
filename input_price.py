import requests
import yfinance as yf
import pandas as pd
import time
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# END_DATE = datetime.now().date()
# Used 2026-02-20 as case study document date. Feel free to change into today's date

END_DATE = datetime.strptime('2026-4-17', '%Y-%m-%d')
START_DATE = END_DATE - timedelta(days=6 * 365)

# Europe natural gas (TTF) price and EU Carbon (EUA) price from yahoo finance
def load_yahoo(ticker, period="5y"):
    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"][ticker]
        volume = df["Volume"][ticker]
    else:
        close = df["Close"]
        volume = df["Volume"]

    out = pd.DataFrame({
        "close": pd.to_numeric(close, errors="coerce"),
        "volume": pd.to_numeric(volume, errors="coerce")
    })

    out.index = pd.to_datetime(out.index)
    out = out.sort_index()

    return out





# Load variables from the .env file
load_dotenv()

# EU Gas Storage from AGSI
def fetch_all_pages(params):
    API_KEY = os.getenv("AGSI_API_KEY")
    BASE_URL = "https://agsi.gie.eu/api"
    REQUEST_DELAY = 1.1
    all_records = []
    page = 1
    headers = {'x-key': API_KEY}
    while True:
        params['page'] = page
        try:
            resp = requests.get(BASE_URL, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"Error on page {page}: {e}")
            break

        records = data.get('data', [])
        if not records:
            break
        all_records.extend(records)

        total_pages = data.get('last_page', 1)
        if page >= total_pages:
            break
        page += 1
        time.sleep(REQUEST_DELAY)

    return all_records

def get_gas_storage():
    params = {
        'type': 'eu',
        'from': START_DATE.isoformat(),
        'to': END_DATE.isoformat(),
        'size': 300
    }

    records = fetch_all_pages(params)
    df = pd.DataFrame(records)

    df['gasDayStart'] = pd.to_datetime(df['gasDayStart'])
    df['updatedAt'] = pd.to_datetime(df['updatedAt'], errors='coerce')

    df = df[['gasDayStart', 'updatedAt', 'gasInStorage', 'full']]
    df = df.sort_values(['gasDayStart', 'updatedAt'])
    df = df.drop_duplicates(subset='gasDayStart', keep='last')

    df = df.rename(columns={
        'gasDayStart': 'date',
        'gasInStorage': 'eu_storage_twh',
        'full': 'eu_storage_pct'
    })

    df = df.set_index('date').sort_index()
    df["eu_storage_pct"] = pd.to_numeric(df["eu_storage_pct"], errors="coerce") # string into numerical
    df["eu_storage_twh"] = pd.to_numeric(df["eu_storage_twh"], errors="coerce")
    df = df.drop(columns=['updatedAt'])

    return df


# Power Daily Price from energy-charts
def get_power_price(bzn="DE-LU"):
    url = "https://api.energy-charts.info/price"
    params = {"bzn": bzn, "start": START_DATE.isoformat(), "end": END_DATE.isoformat()}
    response = requests.get(url, params=params)
    data = response.json()

    df_power = pd.DataFrame({"timestamp": data["unix_seconds"], "Power_DE_DA": data["price"]})
    df_power["Date"] = pd.to_datetime(df_power["timestamp"], unit="s")
    df_power = df_power.set_index("Date")
    df_power = df_power.drop(columns=["timestamp"])
    power_daily = df_power.resample("D").mean()
    power_daily.index = pd.to_datetime(power_daily.index).date
    power_daily.index.name = "date"

    return power_daily



if __name__ == "__main__":
    # print(get_power_price())
    print(get_gas_storage())
    # ttf = load_yahoo("TTF=F")
    # print(ttf.tail())
    # eua = load_yahoo("CO2.L")
    # print(eua.tail())

