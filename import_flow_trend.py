import re
import pandas as pd
import numpy as np
from entsog import EntsogPandasClient



def snake_case(s: str) -> str:
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", s)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.replace("-", "_").lower()


def find_norway_to_eu_entry_points(points: pd.DataFrame) -> pd.DataFrame:
    p = points.copy()
    p.columns = [snake_case(c) for c in p.columns]

    # Build point_direction key used for queries
    p["point_direction"] = p["operator_key"].astype(str) + p["point_key"].astype(str) + p["direction_key"].astype(str)

    norway_mask = p["id"].astype(str).str.contains("NO-TSO-0001", case=False, na=False)

    # entry into EU side
    entry_mask = p["direction_key"].str.contains("entry", case=False, na=False)

    # pipeline interconnection points
    itp_mask = p["point_key"].str.contains("ITP-", case=False, na=False)

    label_blob = (p["point_label"].astype(str) + " " + p["operator_label"].astype(str))
    landing_mask = pd.Series(False, index=p.index)
    LANDING_KEYWORDS = ["Emden", "Dornum", "Zeebrugge", "Dunkerque"]

    for k in LANDING_KEYWORDS:
        landing_mask |= label_blob.str.contains(k, case=False, na=False)

    cand = p.loc[norway_mask & entry_mask & itp_mask & landing_mask,
    ["point_direction", "point_label", "operator_label", "operator_key", "point_key", "direction_key",
     "connected_operators", "id"]].copy()

    return cand.drop_duplicates().sort_values("point_direction")


def fetch_physical_flow_daily(client: EntsogPandasClient,
                              point_directions: list[str],
                              start: pd.Timestamp,
                              end: pd.Timestamp) -> pd.DataFrame:
    """
    Fetch physical_flow and aggregate daily total imports.
    """
    df = client.query_operational_point_data(
        start=start,
        end=end,
        indicators=["physical_flow"],
        point_directions=point_directions,
        verbose=False
    )

    if df is None or df.empty:
        raise RuntimeError("query_operational_point_data returned empty.")

    df2 = df.copy()
    df2.columns = [snake_case(c) for c in df2.columns]

    if isinstance(df2.index, pd.DatetimeIndex):
        df2 = df2.reset_index().rename(columns={"index": "ts"})

    # Identify time/value columns
    time_col = None
    for c in ["period_from", "periodfrom", "ts", "date", "timestamp"]:
        if c in df2.columns:
            time_col = c
            break
    if time_col is None:
        raise KeyError(f"Cannot find time column. Columns={list(df2.columns)}")

    val_col = None
    for c in ["value", "flow", "quantity", "indicator_value", "datavalue"]:
        if c in df2.columns:
            val_col = c
            break
    if val_col is None:
        raise KeyError(f"Cannot find value column. Columns={list(df2.columns)}")

    # Mixed timezones: force parse to UTC, then convert
    dt = pd.to_datetime(df2[time_col], errors="coerce", utc=True)
    dt = dt.dt.tz_convert("Europe/Brussels").dt.normalize()

    df2["date"] = dt
    df2["flow"] = pd.to_numeric(df2[val_col], errors="coerce")

    daily = (df2.groupby("date", as_index=False)["flow"]
             .sum()
             .rename(columns={"flow": "imports_total"}))

    return daily


def fetch_physical_flow_daily_batched(
    client: EntsogPandasClient,
    point_directions: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    batch_size: int = 4,
) -> pd.DataFrame:
    frames = []

    for batch in np.array_split(point_directions, max(1, int(np.ceil(len(point_directions) / batch_size)))):
        batch = [x for x in batch.tolist() if pd.notna(x)]

        df = client.query_operational_point_data(
            start=start,
            end=end,
            indicators=["physical_flow"],
            point_directions=batch,
            verbose=False,
        )

        if df is None or df.empty:
            continue

        df2 = df.copy()
        df2.columns = [snake_case(c) for c in df2.columns]

        if isinstance(df2.index, pd.DatetimeIndex):
            df2 = df2.reset_index().rename(columns={"index": "ts"})

        time_col = next((c for c in ["period_from", "periodfrom", "ts", "date", "timestamp"] if c in df2.columns), None)
        val_col  = next((c for c in ["value", "flow", "quantity", "indicator_value", "datavalue"] if c in df2.columns), None)
        if time_col is None or val_col is None:
            raise KeyError(f"Missing time/value col. cols={list(df2.columns)}")

        dt = pd.to_datetime(df2[time_col], errors="coerce", utc=True).dt.tz_convert("Europe/Brussels").dt.normalize()
        df2["date"] = dt
        df2["flow"] = pd.to_numeric(df2[val_col], errors="coerce")
        frames.append(df2[["date", "flow"]])

    if not frames:
        raise RuntimeError("All batches returned empty (or timed out).")

    all_flows = pd.concat(frames, ignore_index=True)
    daily = (all_flows.groupby("date", as_index=False)["flow"]
             .sum()
             .rename(columns={"flow": "imports_total"}))
    return daily

def compute_trend(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy().sort_values("date").set_index("date")
    d["imports_7d_avg"] = d["imports_total"].rolling(7, min_periods=7).mean()
    d["imports_30d_avg"] = d["imports_total"].rolling(30, min_periods=30).mean()
    d["imports_trend_pct"] = (d["imports_7d_avg"] / d["imports_30d_avg"] - 1.0) * 100.0
    return d.reset_index()



client = EntsogPandasClient()
points = client.query_operator_point_directions()
cand = find_norway_to_eu_entry_points(points)
point_dirs = cand["point_direction"].dropna().unique().tolist()

START = pd.Timestamp("2025-01-01", tz="Europe/Brussels")
END = pd.Timestamp("2026-03-31", tz="Europe/Brussels")
daily = fetch_physical_flow_daily_batched(client, point_dirs, START, END, batch_size=3)
trend = compute_trend(daily)

trend2 = trend.copy().set_index("date")
val = trend2.loc["2026-03-30", "imports_trend_pct"]