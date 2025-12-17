import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.colors import hex_to_rgb
import plotly.express as px

st.set_page_config(page_title="Finetuning Curves", layout="wide")

# ----------------------------
# Helpers
# ----------------------------
def rgba(hex_color: str, alpha: float) -> str:
    r, g, b = hex_to_rgb(hex_color)
    return f"rgba({r},{g},{b},{alpha})"

def prep_series(sub: pd.DataFrame, ycol: str):
    """
    Prepare a (step, y) series for plotting without diagonal connectors.

    - Sort by step
    - Collapse duplicate steps by mean
    - Insert None breaks when step resets
    """
    if sub.empty:
        return [], []

    s = (sub[["step", ycol]]
         .groupby("step", as_index=False)[ycol]
         .mean()
         .sort_values("step"))

    steps = s["step"].tolist()
    vals = s[ycol].tolist()

    xs, ys = [], []
    prev = None
    for x, y in zip(steps, vals):
        if prev is not None and x < prev:
            xs.append(None); ys.append(None)
        xs.append(x); ys.append(y)
        prev = x
    return xs, ys

@st.cache_data
def load_df(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()

    df["pretrain_pct_disp"] = df["pretrain_pct"].apply(
        lambda x: "baseline" if pd.isna(x) else f"{int(x)}%"
    )

    size_order = ["1k", "10k", "100k", "1m"]
    df["size"] = pd.Categorical(df["size"], categories=size_order, ordered=True)

    metric_order = ["loss_train", "loss_val", "acc_val"]
    df["metric"] = pd.Categorical(df["metric"], categories=metric_order, ordered=True)

    return df

# ----------------------------
# Load
# ----------------------------
df = load_df("curves.parquet")

# Remove baseline entirely
df = df[df["pretrain_pct_disp"] != "baseline"].copy()

st.title("Finetune Learning Curves")

# ----------------------------
# Sidebar controls (NO mode selector: always show all modes)
# ----------------------------
st.sidebar.header("Filters")

task_vals = sorted(df["task"].dropna().unique().tolist())
mode_vals = sorted(df["mode"].dropna().unique().tolist())
size_vals = [s for s in df["size"].cat.categories if s in set(df["size"].dropna().unique())]

def pct_sort_key(s: str):
    return int(s[:-1])  # only percentages now

pct_vals = sorted(df["pretrain_pct_disp"].dropna().unique().tolist(), key=pct_sort_key)

task_sel = st.sidebar.selectbox("Task (choose one)", task_vals, index=0, key="task_single")
size_sel = st.sidebar.selectbox("Size (choose one)", size_vals, index=0, key="size_single")

pct_sel = st.sidebar.selectbox(
    "Pretrain % (choose one)",
    pct_vals,
    index=0,
    key="pretrain_pct_single"
)

view_mode = st.sidebar.radio(
    "Metric view",
    ["Accuracy (val)", "Losses (train + val)"],
    index=0,
    key="metric_view"
)

show_bands = st.sidebar.checkbox("Show ± std bands", value=True, key="show_bands")

# ----------------------------
# Determine metrics to plot
# ----------------------------
if view_mode == "Accuracy (val)":
    metrics_to_plot = ["acc_val"]
else:
    metrics_to_plot = ["loss_train", "loss_val"]

# ----------------------------
# Filter + aggregate across trials
# ----------------------------
dff = df[
    (df["task"] == task_sel) &
    (df["size"] == size_sel) &
    (df["pretrain_pct_disp"] == pct_sel) &
    (df["metric"].isin(metrics_to_plot))
].copy()

if dff.empty:
    st.warning("No data matches your filters.")
    st.stop()

group_cols = ["task", "size", "mode", "metric", "step"]
agg = (dff.groupby(group_cols, observed=True)["value"]
         .agg(mean="mean", std="std")
         .reset_index())

agg["std"] = agg["std"].fillna(0)
agg["upper"] = agg["mean"] + agg["std"]
agg["lower"] = agg["mean"] - agg["std"]

# If std is basically all zero, the band will be invisible (it collapses onto the mean line)
if show_bands and float(agg["std"].max()) == 0.0:
    st.info("Std bands are enabled, but std is 0 everywhere (likely only 1 trial per step), so bands collapse onto the mean.")

# ----------------------------
# Single plot (solid lines; loss_val = different color vs loss_train)
# ----------------------------
fig = go.Figure()

palette = px.colors.qualitative.Plotly
alt_palette = px.colors.qualitative.D3

mode_base_color = {m: palette[i % len(palette)] for i, m in enumerate(mode_vals)}
mode_alt_color  = {m: alt_palette[i % len(alt_palette)] for i, m in enumerate(mode_vals)}

for (mode, metric), sub in agg.groupby(["mode", "metric"], observed=True):
    # Color rule:
    # - acc_val / loss_train: base color per mode
    # - loss_val: alternate color per mode
    color = mode_alt_color.get(mode, "#888888") if metric == "loss_val" else mode_base_color.get(mode, "#888888")

    # Bands (make them a bit more visible)
    if show_bands:
        xu, yu = prep_series(sub, "upper")
        xl, yl = prep_series(sub, "lower")

        fig.add_trace(go.Scatter(
            x=xu, y=yu,
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=xl, y=yl,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=rgba(color, 0.22),  # bumped up from 0.12 so you can actually see it
            hoverinfo="skip",
            showlegend=False,
        ))

    # Mean curve (ALL SOLID)
    xm, ym = prep_series(sub, "mean")

    if view_mode == "Accuracy (val)":
        name = f"{mode}"
        legendgroup = f"{mode}"
    else:
        name = f"{mode} · {metric}"
        legendgroup = f"{mode}|{metric}"

    fig.add_trace(go.Scatter(
        x=xm, y=ym,
        mode="lines",
        line=dict(color=color, width=2),
        name=name,
        legendgroup=legendgroup,
        showlegend=True,
    ))

y_title = "accuracy" if view_mode == "Accuracy (val)" else "loss"
fig.update_xaxes(title_text="step")
fig.update_yaxes(title_text=y_title)

fig.update_layout(
    height=650,
    margin=dict(l=30, r=30, t=70, b=30),
    title=f"{task_sel} | size={size_sel} | Pretrain: {pct_sel} | View: {view_mode} "
          f"| Aggregated across trials (mean{' ± std' if show_bands else ''})",
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("Quick counts / sanity"):
    st.write("Filtered rows:", len(dff))
    st.write("Unique runs:", dff["run_dir"].nunique())
    st.write("Modes present:", sorted(dff["mode"].unique().tolist()))
    dup_check = (agg.groupby(["task","size","mode","metric"])["step"]
                   .apply(lambda s: s.duplicated().any()))
    st.write("Any duplicate steps within a curve:", bool(dup_check.any()))
    st.dataframe(
        dff[["task","size","mode","pretrain_pct_disp","trial","metric"]]
          .drop_duplicates()
          .sort_values(["task","size","mode","trial","metric"])
    )
