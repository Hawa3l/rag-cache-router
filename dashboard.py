

import json
from pathlib import Path
import sqlite3

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "logs.db"
QUESTIONS_PATH = BASE_DIR / "questions.json"
TABLE = None  

COLUMN_CANDIDATES = {
    "row_id": ["id", "rowid"],
    "run_id": ["run_id", "batch_id"],
    "timestamp": ["timestamp", "created_at"],
    "question": ["question", "query"],
    "pipeline": ["pipeline"],
    "model": ["model_used", "model"],
    "latency_ms": ["latency_ms", "elapsed_ms"],
    "cost": ["cost_usd", "cost"],
    "prompt_tokens": ["input_tokens", "prompt_tokens"],
    "completion_tokens": ["output_tokens", "completion_tokens"],
    "cache_hit": ["cache_hit"],
    "answer": ["answer", "response"],
}
CACHE_MODEL_NAME = "cache"


@st.cache_data(ttl=15)
def load_data(db_path: str, table):
    con = sqlite3.connect(db_path)
    try:
        if table is None:
            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'",
                con,
            )["name"].tolist()
            if not tables:
                return pd.DataFrame(), None
            table = "logs" if "logs" in tables else tables[0]
        df = pd.read_sql_query(f'SELECT * FROM "{table}"', con)
    finally:
        con.close()

    rename = {}
    for std, options in COLUMN_CANDIDATES.items():
        for opt in options:
            if opt in df.columns:
                rename[opt] = std
                break
    df = df.rename(columns=rename)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    for col in ["prompt_tokens", "completion_tokens", "latency_ms", "cost"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if {"prompt_tokens", "completion_tokens"} <= set(df.columns):
        df["total_tokens"] = df["prompt_tokens"] + df["completion_tokens"]

    if "cache_hit" in df.columns:
        df["is_hit"] = df["cache_hit"].astype(bool)
    elif "model" in df.columns:
        df["is_hit"] = df["model"].astype(str).str.lower() == CACHE_MODEL_NAME
    else:
        df["is_hit"] = False

    return df, table


@st.cache_data(ttl=60)
def load_questions(q_path: str):
    p = Path(q_path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def check_key_facts(answer: str, key_facts: list, q_type: str) -> bool:
    if not isinstance(answer, str) or not answer.strip():
        return False
    if not key_facts:
        return True
    ans_lower = answer.lower()
    if q_type == "unanswerable":
        return any(str(k).lower() in ans_lower for k in key_facts)
    return all(str(k).lower() in ans_lower for k in key_facts)


def enrich_with_questions(df: pd.DataFrame, questions: list) -> pd.DataFrame:
    if not questions or "question" not in df.columns:
        return df

    q_lookup = {}
    for q in questions:
        q_text = q.get("question", "").strip()
        q_lookup.setdefault(q_text, []).append(q)

    df = df.copy()
    q_ids, q_types, q_diffs, q_sources, fact_passes = [], [], [], [], []

    seen_counts = {}
    df_sorted = df.sort_values(by="row_id") if "row_id" in df.columns else df

    for _, row in df_sorted.iterrows():
        p = str(row.get("pipeline", ""))
        q_text = str(row.get("question", "")).strip()
        matches = q_lookup.get(q_text, [])

        if matches:
            idx = seen_counts.get((p, q_text), 0)
            meta = matches[min(idx, len(matches) - 1)]
            seen_counts[(p, q_text)] = idx + 1

            q_ids.append(meta.get("id"))
            q_types.append(meta.get("type"))
            q_diffs.append(meta.get("difficulty"))
            q_sources.append(meta.get("source"))
            passed = check_key_facts(
                str(row.get("answer", "")),
                meta.get("key_facts", []),
                meta.get("type", "original"),
            )
            fact_passes.append(passed)
        else:
            q_ids.append(None)
            q_types.append(None)
            q_diffs.append(None)
            q_sources.append(None)
            fact_passes.append(None)

    df_sorted["q_id"] = q_ids
    df_sorted["q_type"] = q_types
    df_sorted["difficulty"] = q_diffs
    df_sorted["source"] = q_sources
    df_sorted["key_facts_pass"] = fact_passes
    return df_sorted


def p95(s: pd.Series) -> float:
    return float(s.quantile(0.95)) if not s.empty else 0.0


# ---------------------------------------------------------------- page setup
st.set_page_config(page_title="RAG Cache + Router Benchmark", layout="wide")
st.title("RAG Semantic Cache & Cost-Aware Routing")
st.caption(
    "Costs are ESTIMATES calculated at standard list prices per 1M tokens on a 40-question benchmark. "
    "Latencies are measured end-to-end in milliseconds (ms)."
)

if not DB_PATH.exists():
    st.error(
        f"Could not find `{DB_PATH}`. Run this from the project folder after logging requests."
    )
    st.stop()

df_raw, table_used = load_data(str(DB_PATH), TABLE)
if df_raw.empty:
    st.info("`logs.db` has no rows yet. Run `benchmark.py` first.")
    st.stop()

questions_data = load_questions(str(QUESTIONS_PATH))
df = enrich_with_questions(df_raw, questions_data)

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Controls & Filters")
if st.sidebar.button("🔄 Refresh Data", width="stretch"):
    st.cache_data.clear()
    st.rerun()

if "run_id" in df.columns and df["run_id"].notna().any():
    runs = sorted(df["run_id"].dropna().astype(str).unique())
    chosen_runs = st.sidebar.multiselect("Run ID", runs, default=runs)
    df = df[df["run_id"].astype(str).isin(chosen_runs)]

if "pipeline" in df.columns:
    pipelines = sorted(df["pipeline"].dropna().astype(str).unique())
    chosen_p = st.sidebar.multiselect("Pipeline", pipelines, default=pipelines)
    df = df[df["pipeline"].astype(str).isin(chosen_p)]

if "difficulty" in df.columns and df["difficulty"].notna().any():
    diffs = sorted(df["difficulty"].dropna().astype(str).unique())
    chosen_d = st.sidebar.multiselect("Difficulty", diffs, default=diffs)
    df = df[
        df["difficulty"].isin(chosen_d)
        | df["difficulty"].isna()
    ]

keep_latest_40 = st.sidebar.checkbox(
    "Keep only latest 40 rows per pipeline (clean benchmark view)", value=True
)
if keep_latest_40 and "pipeline" in df.columns:
    sort_key = "row_id" if "row_id" in df.columns else "timestamp"
    if sort_key in df.columns:
        df = df.sort_values(sort_key).groupby("pipeline", as_index=False).tail(40)

if df.empty:
    st.info("No rows match the current filters.")
    st.stop()

with st.expander("Detected SQLite Table & Schema (Debug)"):
    st.write(
        f"**Table:** `{table_used}` | **Matched `questions.json`:** `{bool(questions_data)}`"
    )
    st.write("**Columns:**", list(df.columns))

# ---------------------------------------------------------------- top KPIs
st.subheader("Overview (Selected Pipelines)")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Requests (n)", len(df))
if "cost" in df.columns:
    c2.metric("Total Est. Cost", f"${df['cost'].sum():.5f}")
if "latency_ms" in df.columns:
    c3.metric("Median Latency (p50)", f"{df['latency_ms'].median():,.1f} ms")
    c4.metric("p95 Latency", f"{p95(df['latency_ms']):,.1f} ms")
if "is_hit" in df.columns:
    c5.metric(
        "Overall Cache Hit Rate",
        f"{df['is_hit'].mean():.1%}",
        f"{int(df['is_hit'].sum())} hits",
    )

# ---------------------------------------------------------------- per-pipeline table & charts
if "pipeline" in df.columns:
    st.subheader("Pipeline Comparison")
    agg = {"n": ("pipeline", "size")}
    if "is_hit" in df.columns:
        agg["cache_hits"] = ("is_hit", "sum")
        agg["hit_rate"] = ("is_hit", "mean")
    if "latency_ms" in df.columns:
        agg["median_latency_ms"] = ("latency_ms", "median")
        agg["p95_latency_ms"] = ("latency_ms", p95)
    if "total_tokens" in df.columns:
        agg["total_tokens"] = ("total_tokens", "sum")
    if "cost" in df.columns:
        agg["total_cost_usd"] = ("cost", "sum")
        agg["mean_cost_usd"] = ("cost", "mean")
    if "key_facts_pass" in df.columns and df["key_facts_pass"].notna().any():
        agg["quality_pass_rate"] = ("key_facts_pass", "mean")

    summary = df.groupby("pipeline").agg(**agg)
    st.dataframe(
        summary.style.format(
            {
                "hit_rate": "{:.1%}",
                "median_latency_ms": "{:,.1f}",
                "p95_latency_ms": "{:,.1f}",
                "total_tokens": "{:,.0f}",
                "total_cost_usd": "${:.5f}",
                "mean_cost_usd": "${:.6f}",
                "quality_pass_rate": "{:.1%}",
            },
            na_rep="-",
        ),
        width="stretch",
    )

    col_left, col_mid, col_right = st.columns(3)
    if "latency_ms" in df.columns:
        with col_left:
            st.markdown("**Latency (ms): Median vs p95**")
            st.bar_chart(summary[["median_latency_ms", "p95_latency_ms"]])
    if "cost" in df.columns:
        with col_mid:
            st.markdown("**Total Estimated Cost ($ USD)**")
            st.bar_chart(summary[["total_cost_usd"]])
    if "model" in df.columns:
        with col_right:
            st.markdown("**Requests by Model / Cache Tier**")
            model_counts = (
                df.groupby(["pipeline", "model"]).size().unstack(fill_value=0)
            )
            st.bar_chart(model_counts)

# ---------------------------------------------------------------- cache hits vs misses
if "is_hit" in df.columns and "latency_ms" in df.columns and df["is_hit"].any():
    st.subheader("Cache Performance: Hits vs. Misses")
    hm = (
        df.assign(result=df["is_hit"].map({True: "Cache Hit", False: "Cache Miss"}))
        .groupby(["pipeline", "result"])["latency_ms"]
        .agg(
            n="size",
            median_ms="median",
            p95_ms=p95,
            min_ms="min",
            max_ms="max",
        )
        .round(1)
    )
    st.dataframe(hm, width="stretch")
    st.caption(
        "Hits and misses are reported separately so fast local cache hits (~30–90 ms) don't mask LLM tail latency."
    )

# ---------------------------------------------------------------- head-to-head savings vs baseline
if (
    {"pipeline", "question", "cost", "latency_ms"} <= set(df.columns)
    and df["pipeline"].nunique() > 1
):
    st.subheader("Head-to-Head Savings vs. Baseline")
    pipes = sorted(df["pipeline"].unique())
    default_base = "baseline" if "baseline" in pipes else pipes[0]
    c_sel1, c_sel2 = st.columns(2)
    base = c_sel1.selectbox(
        "Baseline Pipeline", pipes, index=pipes.index(default_base)
    )
    other_opts = [p for p in pipes if p != base]
    other = c_sel2.selectbox("Candidate Pipeline", other_opts)

    df_base = df[df["pipeline"] == base]
    df_other = df[df["pipeline"] == other]

    join_col = (
        "q_id"
        if ("q_id" in df.columns and df["q_id"].notna().any())
        else "question"
    )
    a = df_base.groupby(join_col).agg(
        cost=("cost", "sum"),
        latency=("latency_ms", "median"),
        tokens=("total_tokens", "sum"),
    )
    b = df_other.groupby(join_col).agg(
        cost=("cost", "sum"),
        latency=("latency_ms", "median"),
        tokens=("total_tokens", "sum"),
    )
    common = a.index.intersection(b.index)

    if len(common) == 0:
        st.warning("No shared questions found between these two pipelines.")
    else:
        base_cost = a.loc[common, "cost"].sum()
        other_cost = b.loc[common, "cost"].sum()
        base_tokens = a.loc[common, "tokens"].sum()
        other_tokens = b.loc[common, "tokens"].sum()
        base_med_lat = a.loc[common, "latency"].median()
        other_med_lat = b.loc[common, "latency"].median()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Shared Questions Compared", f"{len(common)} questions")
        if base_cost > 0:
            cost_saving = 1.0 - (other_cost / base_cost)
            m2.metric(
                "Cost Reduction",
                f"{cost_saving:.1%}",
                f"-${base_cost - other_cost:.5f}",
            )
        else:
            m2.metric(
                "Cost Reduction",
                "N/A ($0 baseline)",
                "Check PRICES in config.py",
            )

        if base_tokens > 0:
            tok_saving = 1.0 - (other_tokens / base_tokens)
            m3.metric(
                "Token Reduction",
                f"{tok_saving:.1%}",
                f"-{int(base_tokens - other_tokens):,} tokens",
            )

        if other_med_lat > 0:
            lat_change = 1.0 - (other_med_lat / base_med_lat)
            m4.metric(
                "Median Latency Change",
                f"{other_med_lat:,.1f} ms",
                f"{lat_change:+.1%} vs {base}",
            )

# ---------------------------------------------------------------- quality & cache safety audit
st.subheader("Quality & Cache Safety Audit (`questions.json`)")
if "q_type" in df.columns and df["q_type"].notna().any():
    q_col1, q_col2 = st.columns(2)

    with q_col1:
        st.markdown("**Key Facts Accuracy by Pipeline & Difficulty**")
        qual_table = (
            df.dropna(subset=["key_facts_pass"])
            .groupby(["pipeline", "difficulty"])["key_facts_pass"]
            .agg(questions="size", passed="sum", accuracy="mean")
        )
        st.dataframe(
            qual_table.style.format({"accuracy": "{:.1%}"}),
            width="stretch",
        )

    with q_col2:
        st.markdown("**Cache Behavior by Question Type (Repeats, Paraphrases, Near-Misses)**")
        cache_audit = (
            df.groupby(["pipeline", "q_type"])["is_hit"]
            .agg(total="size", hits="sum", hit_rate="mean")
        )
        st.dataframe(
            cache_audit.style.format({"hit_rate": "{:.1%}"}),
            width="stretch",
        )

        near_misses = df[df["q_type"] == "near_miss"]
        if not near_misses.empty:
            false_hits = int(near_misses["is_hit"].sum())
            if false_hits == 0:
                st.success(
                    f"✅ **0 / {len(near_misses)} False Cache Hits on Near-Misses** (100% Cache Precision)"
                )
            else:
                bad_ids = near_misses[near_misses["is_hit"]]["q_id"].tolist()
                st.error(
                    f"⚠️ **{false_hits} False Cache Hit(s) detected on Near-Miss questions:** {bad_ids}"
                )
else:
    st.info(
        "Could not match rows against `questions.json`. Place `questions.json` in the same folder."
    )

# ---------------------------------------------------------------- raw rows
st.subheader("Logged Requests Explorer")
only_failures = st.checkbox("Show only failed `key_facts` checks or false cache hits")
view_df = df.copy()
if only_failures and "key_facts_pass" in view_df.columns:
    view_df = view_df[
        (view_df["key_facts_pass"] == False)
        | ((view_df["q_type"] == "near_miss") & (view_df["is_hit"] == True))
    ]

sort_col = (
    "row_id"
    if "row_id" in view_df.columns
    else ("timestamp" if "timestamp" in view_df.columns else view_df.columns[0])
)
st.dataframe(
    view_df.sort_values(sort_col, ascending=False), width="stretch"
)