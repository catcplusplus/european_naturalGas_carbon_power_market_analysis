from input_price import *
import matplotlib.pyplot as plt

###TTF and EUA front month price and 5 day percentage change
def add_5d_pct_change(df, col_name = "close"):
    df = df.copy()
    df["pct_5d"] = df[col_name].pct_change(5)
    return df

ttf_df = load_yahoo("TTF=F", period="6y")
ttf_df = add_5d_pct_change(ttf_df)

eua_df = load_yahoo("CO2.L", period="6y")
eua_df = add_5d_pct_change(eua_df)



### Implied CCGT marginal cost (€/MWh)
def add_ccgt_cost_column(
    df,
    ttf_col="ttf",
    eua_col="eua",
    efficiency=0.55,
    co2_t_per_mwhth=0.202,
    vom=2.0,
    output_col="ccgt_cost"
):
    """
    Add implied CCGT marginal cost column to dataframe.
    """
    heat_rate = 1 / efficiency
    co2_t_per_mwhe = co2_t_per_mwhth * heat_rate

    df = df.copy()

    df[output_col] = (
        df[ttf_col] * heat_rate +
        df[eua_col] * co2_t_per_mwhe +
        vom
    )

    return df

fuel_df = pd.DataFrame({
    "ttf": ttf_df["close"],
    "eua": eua_df["close"]
}).dropna()

fuel_df = add_ccgt_cost_column(fuel_df)






### plot for power_DE_DA and implied ccgt cost for the past one year
power_price = get_power_price()

power_df = add_5d_pct_change(power_price, "Power_DE_DA")

master_df = fuel_df[["ccgt_cost"]].join(
    power_price[["Power_DE_DA"]],
    how="inner"
)

master_df = master_df.sort_index()
master_df["spark_spread"] = (
    master_df["Power_DE_DA"] - master_df["ccgt_cost"]
)


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "figure.titlesize": 18
})


def plot_power_vs_ccgt(df, years=1, save_path="power_vs_ccgt_1y.pdf"):
    """
    Plot the DE power day ahead price and implied CCGT cost
    Highlight the shade when clean spark spread < 0
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    end_date = df.index.max()
    start_date = end_date - pd.DateOffset(years=years)
    plot_df = df.loc[start_date:end_date]

    fig, ax = plt.subplots(figsize=(12, 6))

    power = plot_df["Power_DE_DA"]
    ccgt = plot_df["ccgt_cost"]

    ax.plot(
        plot_df.index,
        power,
        linewidth=2.2,
        label="DE Day-Ahead Power"
    )

    ax.plot(
        plot_df.index,
        ccgt,
        linewidth=2.2,
        label="Implied CCGT Marginal Cost"
    )

    # Shade region where Power < CCGT
    ax.fill_between(
        plot_df.index,
        power,
        ccgt,
        where=(power < ccgt),
        interpolate=True,
        alpha=0.20
    )

    ax.set_title(
        f"DE Power vs Implied CCGT Cost (Last {years} Year)",
        fontweight="bold"
    )
    ax.set_ylabel("€/MWh")
    ax.set_xlabel("")

    ax.margins(x=0)
    ax.set_xlim(plot_df.index.min(), plot_df.index.max())

    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


# plot_power_vs_ccgt(master_df)