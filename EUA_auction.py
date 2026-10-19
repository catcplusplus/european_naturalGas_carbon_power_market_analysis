import io
import re
import requests
import pandas as pd
from urllib.parse import urljoin
from input_price import load_yahoo

INDEX_URL = "https://public.eex-group.com/eex/eua-auction-report/index.html"

def get_latest_eex_auction_xlsx_url(index_url: str = INDEX_URL) -> str:
    html = requests.get(index_url, timeout=60).text
    hrefs = re.findall(r'href="([^"]+\.xlsx)"', html, flags=re.IGNORECASE)

    data_hrefs = [
        h for h in hrefs
        if "emission-spot-primary-market-auction-report" in h.lower()
           and h.lower().endswith("-data.xlsx")
    ]
    if not data_hrefs:
        raise RuntimeError("No '*-data.xlsx' links found on the EEX auction report index page.")

    def get_year(h: str) -> int:
        m = re.search(r"report-(\d{4})-data\.xlsx", h, flags=re.IGNORECASE)
        return int(m.group(1)) if m else -1

    best_href = sorted(data_hrefs, key=get_year)[-1]
    return urljoin(index_url, best_href)


def download_xlsx(url: str) -> bytes:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content


def find_header_row(df_noheader: pd.DataFrame) -> int:
    """
    Find the header row by locating a row that contains both:
    'Date' and 'Auction Volume'
    """
    max_scan = min(80, len(df_noheader))

    for i in range(max_scan):
        row_cells = []
        for v in df_noheader.iloc[i].tolist():
            if pd.isna(v):
                row_cells.append("")
            else:
                row_cells.append(str(v).strip().lower())

        has_date = any(c == "date" or c.startswith("date") for c in row_cells)
        has_volume = any("auction volume" in c for c in row_cells)

        if has_date and has_volume:
            return i

    raise RuntimeError("Could not locate header row containing 'Date' and 'Auction Volume'.")


def load_primary_market_auction_table(xlsx_bytes: bytes) -> pd.DataFrame:
    xls = pd.ExcelFile(io.BytesIO(xlsx_bytes))
    if "Primary Market Auction" not in xls.sheet_names:
        raise RuntimeError(f"Expected sheet 'Primary Market Auction'. Found: {xls.sheet_names}")

    df0 = pd.read_excel(xls, sheet_name="Primary Market Auction", header=None)
    header_row = find_header_row(df0)

    df = pd.read_excel(xls, sheet_name="Primary Market Auction", header=header_row)
    return df


def normalize_eex_auction_df(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Date", "Auction Price €/tCO2", "Auction Volume tCO2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}. Columns found: {list(df.columns)}")

    out = df.copy()

    out["date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()

    # Price + volume
    out["auction_price_eur_t"] = pd.to_numeric(out["Auction Price €/tCO2"], errors="coerce")
    out["auction_volume_tco2"] = pd.to_numeric(out["Auction Volume tCO2"], errors="coerce")

    # Cover ratio
    if "Cover Ratio" in out.columns:
        out["cover_ratio"] = pd.to_numeric(out["Cover Ratio"], errors="coerce")
    else:
        out["cover_ratio"] = pd.NA

    out = out.dropna(subset=["date", "auction_volume_tco2"])
    out = out.sort_values("date")

    return out[["date", "auction_price_eur_t", "auction_volume_tco2", "cover_ratio"]]


def compute_auction_indicators(auction, how='inner'):
    eua = load_yahoo("CO2.L", period="1y")
    auction = auction.copy()

    eua.index = pd.to_datetime(eua.index)
    auction['date'] = pd.to_datetime(auction['date'])

    merged = eua.merge(
        auction,
        left_index=True,
        right_on='date',
        how=how
    )
    merged = merged.set_index('date')

    merged['auction_premium_pct'] = (merged['auction_price_eur_t'] - merged['close']) / merged['close'] * 100

    merged['cover_prev5_mean'] = (
        merged['cover_ratio']
        .rolling(5, min_periods=5)
        .mean()
        .shift(1)
    )
    merged["cover_vs_prev5_pct"] = (
            (merged["cover_ratio"] / merged["cover_prev5_mean"] - 1.0) * 100
    )

    return merged

# -- Download the latest EUA auction data
url = get_latest_eex_auction_xlsx_url()
xlsx_bytes = download_xlsx(url)

table = load_primary_market_auction_table(xlsx_bytes)
auctions = normalize_eex_auction_df(table)

auction_and_price = compute_auction_indicators(auctions)
my_case_study = auction_and_price.loc['2026-03-30']