import os
import re
import pandas as pd
import numpy as np
from entsoe import EntsoePandasClient
import time
import random
import requests
from dotenv import load_dotenv


ZONE = "DE_LU"

PSR_SOLAR = "B16"
PSR_WIND_OFF = "B18"
PSR_WIND_ON  = "B19"

START = pd.Timestamp("2026-01-01", tz="Europe/Brussels")
END   = pd.Timestamp("2026-03-31", tz="Europe/Brussels")

load_dotenv()
Entsoe_API_KEY = os.getenv("Entsoe_API_KEY")
client = EntsoePandasClient(api_key=Entsoe_API_KEY)


def yearly_windows(start: pd.Timestamp, end: pd.Timestamp):
    y0 = start.year
    y1 = end.year
    for y in range(y0, y1 + 1):
        s = max(start, pd.Timestamp(f"{y}-01-01", tz=start.tz))
        e = min(end,   pd.Timestamp(f"{y}-12-31 23:00", tz=end.tz) + pd.Timedelta(hours=1))
        if s < e:
            yield s, e


def to_series(x, name: str) -> pd.Series:
    """
    Convert entsoe-py result to a simple Series with DatetimeIndex.
    """
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            s = x.iloc[:, 0]
        else:
            s = x.sum(axis=1)
    else:
        s = x

    # If MultiIndex, try to use the datetime level
    if isinstance(s.index, pd.MultiIndex):
        # first level is timestamp
        dt_level = s.index.get_level_values(0)
        s = pd.Series(s.values, index=pd.to_datetime(dt_level), name=name)

    if s.index.tz is None:
        s.index = s.index.tz_localize("Europe/Brussels")
    else:
        s = s.tz_convert("Europe/Brussels")

    s = s.sort_index()
    s.name = name
    return s


def query_generation_yearly(zone: str, psr_type: str, name: str) -> pd.Series:
    parts = []
    for s, e in yearly_windows(START, END):
        g = client.query_generation(zone, start=s, end=e, psr_type=psr_type)
        parts.append(to_series(g, name))
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out



def query_load_yearly(zone: str) -> pd.Series:
    parts = []
    for s, e in yearly_windows(START, END):
        l = client.query_load(zone, start=s, end=e)
        parts.append(to_series(l, "load_mw"))
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


# ---- Fetch series  ----
load = query_load_yearly(ZONE)
solar = query_generation_yearly(ZONE, PSR_SOLAR, "solar_mw")
wind_on = query_generation_yearly(ZONE, PSR_WIND_ON, "wind_on_mw")
wind_off = query_generation_yearly(ZONE, PSR_WIND_OFF, "wind_off_mw")


df = pd.concat([load, solar, wind_on, wind_off], axis=1, join="inner").sort_index()
df["residual_mw"] = df["load_mw"] - (df["solar_mw"].fillna(0) + df["wind_on_mw"].fillna(0) + df["wind_off_mw"].fillna(0))


# Daily mean residual
daily = df["residual_mw"].resample("D").mean().to_frame("residual_mean")

# Rolling 5-day baseline
daily["mean_5d"] = daily["residual_mean"].rolling(5).mean().shift(1)
daily["std_5d"] = daily["residual_mean"].rolling(5).std(ddof=0).shift(1)

# Z-score
daily["residual_z5"] = (
    (daily["residual_mean"] - daily["mean_5d"]) / daily["std_5d"]
)