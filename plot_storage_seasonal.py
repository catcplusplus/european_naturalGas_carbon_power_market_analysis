from input_price import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

storage_df = get_gas_storage()
latest_storage = storage_df["eu_storage_pct"].iloc[-1]

storage_df["storage_wow_pp"] = (
        storage_df["eu_storage_pct"] -
        storage_df["eu_storage_pct"].shift(7)
)

latest_wow = storage_df["storage_wow_pp"].iloc[-1]


## SEASONAL PLOT##
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "figure.titlesize": 18
})


def plot_storage_seasonal_band(
        storage: pd.DataFrame,
        value_col: str = "eu_storage_pct",
        band_years=(2021, 2022, 2023, 2024, 2025),
        line_year: int = 2026,
        pdf_path: str = "eu_storage_seasonal.pdf",
        title: str = "EU Gas Storage vs 5Y Seasonal Band",
        show: bool = True,
):
    """
      - Band: mean ± 1 std across band_years for each calendar day
      - Line: value_col for line_year 2026
    """
    df = storage.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    df["year"] = df.index.year
    df["month"] = df.index.month
    df["day"] = df.index.day
    df["mmdd"] = df["month"].astype(str).str.zfill(2) + "-" + df["day"].astype(str).str.zfill(2)

    # Remove Feb 29 for consistent 365-day seasonal alignment
    df = df[df["mmdd"] != "02-29"].copy()

    # Dummy seasonal date (non-leap year) for x-axis formatting
    df["seasonal_date"] = pd.to_datetime("2001-" + df["mmdd"], format="%Y-%m-%d")

    hist = df[df["year"].isin(list(band_years))].copy()

    clim = (
        hist.groupby("seasonal_date")[value_col]
        .agg(avg="mean", std="std")
        .sort_index()
    )

    clim["low"] = clim["avg"] - clim["std"]
    clim["high"] = clim["avg"] + clim["std"]

    latest_date = df.index[-1]
    latest_value = df[value_col].iloc[-1]
    latest_mmdd = latest_date.strftime("%m-%d")

    fivey_avg = clim.loc[pd.to_datetime("2001-" + latest_mmdd), "avg"]
    diff = latest_value - fivey_avg

    direction = "above" if diff > 0 else "below"

    sentence = (
        f"Storage is {abs(diff):.1f}pp {direction} 5Y average on the same date."
    )

    print(sentence)

    line_df = df[df["year"] == line_year].copy()
    line_df = line_df.sort_values("seasonal_date")

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.fill_between(
        clim.index,
        clim["low"],
        clim["high"],
        alpha=0.25,
        label=f"{len(band_years)}Y average ± 1 stdev"
    )

    ax.plot(clim.index, clim["avg"], linewidth=2, label=f"{len(band_years)}Y average")

    ax.plot(
        line_df["seasonal_date"],
        line_df[value_col],
        linewidth=2,
        label=str(line_year)
    )

    # Month labels
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_xlim(pd.Timestamp("2001-01-01"), pd.Timestamp("2001-12-31"))

    ax.set_ylabel("Storage (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    # Save to PDF
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return diff


diff = plot_storage_seasonal_band(
    storage_df,
    band_years=(2021, 2022, 2023, 2024, 2025),
    line_year=2026,
    pdf_path="eu_storage_seasonal_5y_band_6.pdf",
    show=False
)