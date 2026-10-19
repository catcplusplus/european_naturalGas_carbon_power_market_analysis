from get_daily_metrics import metrics_df
import json, hashlib, datetime as dt, subprocess
from pathlib import Path
import pandas as pd

OUT_DIR = Path("out")
LOG_DIR = OUT_DIR / "logs"
OUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


def _utc_now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def _stable_hash(obj) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def _df_to_facts(core_df: pd.DataFrame) -> str:
    rows = []
    for _, r in core_df.iterrows():
        rows.append(f"- {str(r['Metric']).strip()}: {float(r['Value']):.2f}")
    return "\n".join(rows)

def ollama_qwen_brief(core_df: pd.DataFrame, model: str = "qwen2.5:3b") -> str:
    """
    Generates a trading narrative using ONLY the metrics in core_df.
    Writes logs: out/logs/llm_<run_id>.json
    Writes output: out/daily_brief_<run_id>.md
    """
    facts = _df_to_facts(core_df)

    required_metrics = [
        "EU gas storage (%)",
        "WoW storage change (pp)",
        "Seasonal deviation (pp vs 5y avg)",
        "Pipeline Imports Trend(7D vs 30D)",
        "TTF (€/MWh)",
        "EUA (€/t)",
        "DE Day-Ahead (€/MWh)",
        "Residual Load Z-Score (5D)",
        "Clean spark spread (€/MWh)",
    ]

    must_mention = "\n".join([f"- {x}" for x in required_metrics])

    prompt = f"""You are a senior European energy trading analyst. Write a concise morning desk note for gas and power traders.

    RULES:
    - Use ONLY the numbers below. Do NOT invent any data.
    - Every statement must cite a metric name and its value.
    - Focus on cross-commodity links: gas → carbon → power.

    FACTS:
    {facts}

    FORMAT:

    Gas: 2-3 sentences on storage stress (level, weekly draw, seasonal deviation), pipeline trend, and TTF implication.

    Carbon: 1-2 sentences linking EUA level and auction premium to gas-driven demand or speculative tone.

    Power: 2-3 sentences on DA vs CCGT cost (spark spread), residual load, and what this means for prompt vs curve risk.

    Bias: One sentence — Upside/Downside/Neutral and the key driver.

    Watch: 2 items that would change the view.""".strip()

    # Call Ollama
    p = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False
    )
    output = p.stdout.decode("utf-8", errors="replace").strip()
    err = p.stderr.decode("utf-8", errors="replace").strip()

    # Log everything for auditability
    payload = {
        "timestamp_utc": _utc_now(),
        "model": model,
        "prompt": prompt,
        "output": output,
        "stderr": err,
        "returncode": p.returncode,
        "metrics": core_df.to_dict(orient="records"),
    }
    run_id = _stable_hash(payload)

    (LOG_DIR / f"llm_{run_id}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    md = output if output else "(No LLM output returned — check Ollama stderr in the log.)"
    (OUT_DIR / f"daily_brief_{run_id}.md").write_text(md, encoding="utf-8")

    return md




if __name__ == "__main__":
    CORE_METRICS = [
        "EU gas storage (%)",
        "Seasonal deviation (pp vs 5y avg)",
        "WoW storage change (pp)",
        "Pipeline Imports Trend(7D vs 30D)",
        "TTF (€/MWh)",
        "EUA (€/t)",
        'Auction price premium in %: ',
        "DE Day-Ahead (€/MWh)",
        "Residual Load Z-Score (5D)",
        "Clean spark spread (€/MWh)",
    ]

    core_df = (
        metrics_df.copy()
        .assign(Metric=lambda x: x["Metric"].astype(str).str.strip())
        .loc[lambda x: x["Metric"].isin(CORE_METRICS), ["Metric", "Value"]]
        .reset_index(drop=True)
    )
    print(core_df)
    brief = ollama_qwen_brief(core_df, model="qwen2.5:3b")
    print(brief)
