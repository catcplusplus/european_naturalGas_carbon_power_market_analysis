from __future__ import annotations
from import_flow_trend import *
from DE_residual_load import daily
from plot_storage_seasonal import *
from EUA_auction import my_case_study
import pandas as pd
from TTF_EUA import *
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np
import json
from dataclasses import asdict

metrics_df = pd.DataFrame({
    "Metric": [
        "EU gas storage (%)",
        "WoW storage change (pp)",
        "Seasonal deviation (pp vs 5y avg)",
        "Pipeline Imports Trend(7D vs 30D)",
        "TTF (€/MWh)",
        "TTF 5d pct",
        "EUA (€/t)",
        "EUA 5d pct",
        'Current cover ratio',
        'Current cover ratio vs previous 5 mean in %: ',
        'Auction price premium in %: ',
        "DE Day-Ahead (€/MWh)",
        "DE 5d pct",
        "DE Residual Load (Daily Mean)",
        "Residual Load Z-Score (5D)",
        "Implied CCGT cost (€/MWh)",
        "Clean spark spread (€/MWh)"

    ],
    "Value": [
        round(latest_storage, 2),
        round(latest_wow, 2),
        round(diff, 2),
        round(val, 2),
        round(ttf_df.loc["2026-03-30"]["close"], 2),
        round(ttf_df.loc["2026-03-30"]["pct_5d"], 2),
        round(eua_df.loc["2026-03-30"]["close"], 2),
        round(eua_df.loc["2026-03-30"]["pct_5d"], 2),
        my_case_study['cover_ratio'],
        round(my_case_study['cover_vs_prev5_pct'], 2),
        round(my_case_study['auction_premium_pct'], 2),
        round(power_df["Power_DE_DA"].iloc[-1], 2),
        round(power_df["pct_5d"].iloc[-1], 2),
        round(daily['residual_mean'].iloc[-1], 2),
        round(daily['residual_z5'].iloc[-1], 2),
        round(master_df['ccgt_cost'].iloc[-1], 2),
        round(master_df["spark_spread"].iloc[-1], 2)
    ]
})

print(metrics_df)

# Data structures
@dataclass
class Signal:
    bucket: str  # "tight/neutral/loose" etc.
    score: int  # 0-100
    reasons: List[str]  # audit trail


@dataclass
class DeskView:
    gas: Signal
    carbon: Signal
    power: Signal
    curve_call: str  # "bull-flatten" etc.
    trade_bias: str  # one-liner
    watchlist: List[str]  # next 24-72h
    key_levels: Dict[str, Any]  # quick access to important levels
    metrics_table: List[Dict[str, Any]]


def normalize_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    if not {"Metric", "Value"}.issubset(df.columns):
        raise ValueError("metrics must have columns ['Metric','Value']")
    out = df.copy()
    out["Metric"] = out["Metric"].astype(str).str.strip()
    out["Value"] = pd.to_numeric(out["Value"], errors="coerce")
    out = out.dropna(subset=["Metric"])
    return out


def df_to_map(df: pd.DataFrame) -> Dict[str, float]:
    m: Dict[str, float] = {}
    for _, r in df.iterrows():
        if pd.notna(r["Value"]):
            m[str(r["Metric"]).strip()] = float(r["Value"])
    return m


def get(m: Dict[str, float], key: str, default=np.nan) -> float:
    return float(m.get(key, default))


def clamp_score(x: float) -> int:
    return int(np.clip(x, 0, 100))


# Trader rule engine
def score_gas(m: Dict[str, float]) -> Signal:
    reasons: List[str] = []
    score = 50

    storage = get(m, "EU gas storage (%)")
    wow_pp = get(m, "WoW storage change (pp)")
    seas_dev = get(m, "Seasonal deviation (pp vs 5y avg)")
    pipe_trend = get(m, "Pipeline Imports Trend(7D vs 30D)")
    ttf = get(m, "TTF (€/MWh)")
    ttf_5d = get(m, "TTF 5d pct")
    cover_vs_5y = get(m, "Current cover ratio vs previous 5 mean in %:")

    # Structural tightness
    if not np.isnan(seas_dev):
        if seas_dev <= -15:
            score += 20;
            reasons.append(f"Seasonal deficit very large ({seas_dev:.2f}pp <= -15pp)")
        elif seas_dev <= -10:
            score += 12;
            reasons.append(f"Seasonal deficit large ({seas_dev:.2f}pp <= -10pp)")
        elif seas_dev >= +10:
            score -= 12;
            reasons.append(f"Seasonal surplus ({seas_dev:.2f}pp >= +10pp)")
        else:
            reasons.append(f"Seasonal deviation modest ({seas_dev:.2f}pp)")

    # Absolute storage level
    if not np.isnan(storage):
        if storage < 35:
            score += 15;
            reasons.append(f"Storage low ({storage:.2f}% < 35%)")
        elif storage > 55:
            score -= 10;
            reasons.append(f"Storage high ({storage:.2f}% > 55%)")
        else:
            reasons.append(f"Storage mid ({storage:.2f}%)")

    # Weekly momentum
    if not np.isnan(wow_pp):
        if wow_pp <= -2:
            score += 10;
            reasons.append(f"Weekly draw strong ({wow_pp:.2f}pp <= -2pp)")
        elif wow_pp >= +2:
            score -= 8;
            reasons.append(f"Weekly injection strong ({wow_pp:.2f}pp >= +2pp)")
        else:
            reasons.append(f"Weekly change mild ({wow_pp:.2f}pp)")

    # Supply trend (pipes)
    if not np.isnan(pipe_trend):
        if pipe_trend <= -0.5:
            score += 8;
            reasons.append(f"Pipeline trend worsening ({pipe_trend:.2f} <= -0.5)")
        elif pipe_trend >= +0.5:
            score -= 6;
            reasons.append(f"Pipeline trend improving ({pipe_trend:.2f} >= +0.5)")
        else:
            reasons.append(f"Pipeline trend stable ({pipe_trend:.2f})")

    # Price & short momentum
    if not np.isnan(ttf):
        if ttf >= 40:
            score += 10;
            reasons.append(f"TTF elevated (€{ttf:.2f} >= 40)")
        elif ttf <= 25:
            score -= 8;
            reasons.append(f"TTF low (€{ttf:.2f} <= 25)")
        else:
            reasons.append(f"TTF mid (€{ttf:.2f})")

    if not np.isnan(ttf_5d):
        if ttf_5d >= 0.05:
            score += 6;
            reasons.append(f"TTF 5d up ({ttf_5d:.2%} >= +5%)")
        elif ttf_5d <= -0.05:
            score -= 5;
            reasons.append(f"TTF 5d down ({ttf_5d:.2%} <= -5%)")
        else:
            reasons.append(f"TTF 5d mild ({ttf_5d:.2%})")

    # Coverage proxy dampens short squeeze probability
    if not np.isnan(cover_vs_5y):
        if cover_vs_5y >= 25:
            score -= 6;
            reasons.append(f"Cover vs 5y high (+{cover_vs_5y:.2f}%) dampens squeeze risk")
        elif cover_vs_5y <= -25:
            score += 6;
            reasons.append(f"Cover vs 5y low ({cover_vs_5y:.2f}%) increases vulnerability")
        else:
            reasons.append(f"Cover vs 5y near normal ({cover_vs_5y:.2f}%)")

    score = clamp_score(score)
    bucket = "tight" if score >= 65 else "loose" if score <= 35 else "neutral"
    return Signal(bucket=bucket, score=score, reasons=reasons)


def score_carbon(m: Dict[str, float]) -> Signal:
    reasons: List[str] = []
    score = 50

    eua = get(m, "EUA (€/t)")
    eua_5d = get(m, "EUA 5d pct")
    auction_prem = get(m, "Auction price premium in %:")

    if not np.isnan(eua):
        if eua >= 80:
            score += 10;
            reasons.append(f"EUA elevated (€{eua:.2f} >= 80)")
        elif eua <= 55:
            score -= 8;
            reasons.append(f"EUA soft (€{eua:.2f} <= 55)")
        else:
            reasons.append(f"EUA mid (€{eua:.2f})")

    if not np.isnan(eua_5d):
        if eua_5d >= 0.03:
            score += 10;
            reasons.append(f"EUA momentum positive ({eua_5d:.2%} >= +3%)")
        elif eua_5d <= -0.03:
            score -= 8;
            reasons.append(f"EUA momentum negative ({eua_5d:.2%} <= -3%)")
        else:
            reasons.append(f"EUA momentum mild ({eua_5d:.2%})")

    if not np.isnan(auction_prem):
        if auction_prem >= 2:
            score += 8;
            reasons.append(f"Auction premium +{auction_prem:.2f}% signals bid tone")
        elif auction_prem <= -2:
            score -= 6;
            reasons.append(f"Auction discount {auction_prem:.2f}% signals offered tone")
        else:
            reasons.append(f"Auction premium near flat ({auction_prem:.2f}%)")

    score = clamp_score(score)
    bucket = "bid" if score >= 60 else "offered" if score <= 40 else "balanced"
    return Signal(bucket=bucket, score=score, reasons=reasons)


def score_power(m: Dict[str, float]) -> Signal:
    reasons: List[str] = []
    score = 50

    da = get(m, "DE Day-Ahead (€/MWh)")
    da_5d = get(m, "DE 5d pct")
    resid_z = get(m, "Residual Load Z-Score (5D)")
    ccgt = get(m, "Implied CCGT cost (€/MWh)")
    css = get(m, "Clean spark spread (€/MWh)")

    # DA vs implied CCGT cost: premium vs fuel-following
    if not np.isnan(da) and not np.isnan(ccgt):
        delta = da - ccgt
        if delta >= 5:
            score += 10;
            reasons.append(f"DA above CCGT cost by +€{delta:.2f} (premium/scarcity)")
        elif delta <= -5:
            score -= 8;
            reasons.append(f"DA below CCGT cost by €{delta:.2f} (soft demand/oversupply)")
        else:
            reasons.append(f"DA ~ fuel-following (DA-CCGT = €{delta:.2f})")

    # Spark margin
    if not np.isnan(css):
        if css >= 8:
            score += 8;
            reasons.append(f"Clean spark wide (€{css:.2f} >= 8)")
        elif css <= 2:
            score -= 4;
            reasons.append(f"Clean spark thin (€{css:.2f} <= 2)")
        else:
            reasons.append(f"Clean spark moderate (€{css:.2f})")

    # Residual load
    if not np.isnan(resid_z):
        if resid_z >= 1:
            score += 8;
            reasons.append(f"Residual load high (Z={resid_z:.2f})")
        elif resid_z <= -1:
            score -= 6;
            reasons.append(f"Residual load low (Z={resid_z:.2f})")
        else:
            reasons.append(f"Residual load near normal (Z={resid_z:.2f})")

    # Momentum
    if not np.isnan(da_5d):
        if da_5d >= 0.05:
            score += 6;
            reasons.append(f"DA 5d up ({da_5d:.2%} >= +5%)")
        elif da_5d <= -0.05:
            score -= 5;
            reasons.append(f"DA 5d down ({da_5d:.2%} <= -5%)")
        else:
            reasons.append(f"DA 5d mild ({da_5d:.2%})")

    score = clamp_score(score)
    bucket = "firm" if score >= 60 else "soft" if score <= 40 else "range"
    return Signal(bucket=bucket, score=score, reasons=reasons)


def infer_curve_call(m: Dict[str, float], gas: Signal, carbon: Signal, power: Signal) -> Tuple[str, str, List[str]]:
    watch: List[str] = []
    seas_dev = get(m, "Seasonal deviation (pp vs 5y avg)")
    pipe_trend = get(m, "Pipeline Imports Trend(7D vs 30D)")
    ttf_5d = get(m, "TTF 5d pct")
    eua_5d = get(m, "EUA 5d pct")
    css = get(m, "Clean spark spread (€/MWh)")

    # Curve logic
    if gas.bucket == "tight" and carbon.bucket != "offered":
        curve_call = "bull-flatten (front supported; season risk premium sticky)"
        trade_bias = "Upside skew in prompt/season; fade downside unless deficit improves"
    elif gas.bucket == "loose" and power.bucket in {"soft", "range"}:
        curve_call = "bear-steepen (front eases; back steadier)"
        trade_bias = "Downside skew in prompt; sell rallies unless new tightness catalyst"
    else:
        curve_call = "range / mixed (wait for catalyst)"
        trade_bias = "Neutral: catalyst-driven (flows/EUA impulse/residual load swing)"

    # Watchlist
    if not np.isnan(seas_dev):
        watch.append(f"Storage seasonal deviation: {seas_dev:.2f}pp (watch refill/draw pace)")
    if not np.isnan(pipe_trend):
        watch.append(f"Pipeline trend (7D vs 30D): {pipe_trend:.2f} (watch step-change)")
    if not np.isnan(ttf_5d):
        watch.append(f"TTF 5d: {ttf_5d:.2%} (watch LNG/pipes surprises)")
    if not np.isnan(eua_5d):
        watch.append(f"EUA 5d: {eua_5d:.2%} (watch auction clears/policy tone)")
    if not np.isnan(css):
        watch.append(f"Clean spark: €{css:.2f} (watch for vol if RE drops)")

    return curve_call, trade_bias, watch[:3]


def build_desk_view(metrics_df: pd.DataFrame) -> DeskView:
    df = normalize_metrics_df(metrics_df)
    m = df_to_map(df)

    gas = score_gas(m)
    carbon = score_carbon(m)
    power = score_power(m)
    curve_call, trade_bias, watchlist = infer_curve_call(m, gas, carbon, power)

    key_levels = {
        "TTF (€/MWh)": get(m, "TTF (€/MWh)"),
        "EUA (€/t)": get(m, "EUA (€/t)"),
        "DE Day-Ahead (€/MWh)": get(m, "DE Day-Ahead (€/MWh)"),
        "Implied CCGT cost (€/MWh)": get(m, "Implied CCGT cost (€/MWh)"),
        "Clean spark spread (€/MWh)": get(m, "Clean spark spread (€/MWh)"),
        "EU gas storage (%)": get(m, "EU gas storage (%)"),
        "Seasonal deviation (pp vs 5y avg)": get(m, "Seasonal deviation (pp vs 5y avg)"),
    }

    metrics_table = (
        df.dropna(subset=["Value"])
        .assign(Value=lambda x: x["Value"].astype(float))
        .to_dict(orient="records")
    )

    return DeskView(
        gas=gas,
        carbon=carbon,
        power=power,
        curve_call=curve_call,
        trade_bias=trade_bias,
        watchlist=watchlist,
        key_levels=key_levels,
        metrics_table=metrics_table,
    )


# view = build_desk_view(metrics_df=metrics_df)
# print(json.dumps(asdict(view), indent=2, ensure_ascii=False))