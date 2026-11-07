"""Warehouse Executive Monitor: a Databricks App (Streamlit) over the gold tables
built by warehouse_exec_data_build.py."""

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

CATALOG = os.getenv("CATALOG", "workspace")
SCHEMA = os.getenv("SCHEMA", "warehouse_exec")
T = f"{CATALOG}.{SCHEMA}"

st.set_page_config(page_title="Warehouse Executive Monitor", page_icon="📦", layout="wide")


# ---------------------------------------------------------------- data access
@st.cache_resource
def get_connection():
    from databricks import sql
    from databricks.sdk.core import Config

    cfg = Config()  # picks up the app service principal automatically
    host = cfg.host.replace("https://", "").replace("http://", "")
    return sql.connect(
        server_hostname=host,
        http_path=f"/sql/1.0/warehouses/{os.environ['DATABRICKS_WAREHOUSE_ID']}",
        credentials_provider=lambda: cfg.authenticate,
    )


def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    """SQL DECIMAL results arrive as Python Decimal objects; turn them into floats so charts and math work."""
    from decimal import Decimal

    for c in df.columns:
        if df[c].dtype == object:
            vals = df[c].dropna()
            if len(vals) and all(isinstance(v, (Decimal, int, float)) for v in vals.head(50)):
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df


@st.cache_data(ttl=600, show_spinner="Loading data...")
def query(sql_text: str) -> pd.DataFrame:
    for attempt in range(2):
        try:
            with get_connection().cursor() as cur:
                cur.execute(sql_text)
                return _numeric(cur.fetchall_arrow().to_pandas())
        except Exception:
            if attempt == 1:
                raise
            get_connection.clear()  # stale connection: reconnect once


# KPI definitions: (label, numerator, denominator, scale, target code, higher is better, format)
KPIS = {
    "OTIF %": ("orders_otif", "orders_shipped", 100, "OTIF", True, "{:.1f}%"),
    "On-time ship %": ("orders_on_time", "orders_shipped", 100, "ON_TIME", True, "{:.1f}%"),
    "Fill rate %": ("units_shipped", "units_ordered_on_shipped", 100, "FILL_RATE", True, "{:.2f}%"),
    "Dock-to-stock (hrs)": ("dock_to_stock_hours_total", "receipts", 1, "DOCK_TO_STOCK", False, "{:.1f}"),
    "Storage utilization %": ("pallets_occupied", "pallet_capacity", 100, "UTILIZATION", False, "{:.1f}%"),
    "Inventory accuracy %": ("cycle_counts_accurate", "cycle_counts", 100, "INV_ACCURACY", True, "{:.2f}%"),
    "Pick units / hour": ("pick_units", "pick_hours", 1, "PICK_UPH", True, "{:.0f}"),
    "Overtime %": ("overtime_hours", "labor_hours", 100, "OVERTIME_PCT", False, "{:.1f}%"),
    "Labor cost / unit": ("labor_cost_usd", "units_shipped", 1, "LABOR_COST_PER_UNIT", False, "${:.2f}"),
    "Orders shipped": ("orders_shipped", None, 1, None, True, "{:,.0f}"),
    "Open backlog (orders)": ("open_backlog_orders", None, 1, None, False, "{:,.0f}"),
}
STATUS_COLORS = {"Green": "#2e9e5b", "Amber": "#e0a100", "Red": "#d64545", "No Data": "#888888"}

warehouses = query(f"SELECT warehouse_id, warehouse_name FROM {T}.dim_warehouse ORDER BY warehouse_id")
targets = query(f"SELECT * FROM {T}.dim_kpi_target")
events = query(f"SELECT * FROM {T}.dim_operational_event ORDER BY start_date")
bounds = query(f"SELECT MIN(kpi_date) AS lo, MAX(kpi_date) AS hi FROM {T}.gold_site_daily_kpis")
min_d, max_d = pd.to_datetime(bounds.lo[0]).date(), pd.to_datetime(bounds.hi[0]).date()
name_of = dict(zip(warehouses.warehouse_id, warehouses.warehouse_name))


def target_for(code, wh="ALL"):
    t = targets[(targets.kpi_code == code) & (targets.warehouse_id == wh)]
    if t.empty:
        t = targets[(targets.kpi_code == code) & (targets.warehouse_id == "ALL")]
    return None if t.empty else float(t.target_value.iloc[0])


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📦 Warehouse Monitor")
    sites = st.multiselect("Sites", warehouses.warehouse_id, default=list(warehouses.warehouse_id),
                           format_func=lambda w: name_of[w])
    date_range = st.date_input("Trend window", (max(min_d, max_d - pd.Timedelta(days=120)), max_d),
                               min_value=min_d, max_value=max_d)
    if st.button("Refresh data"):
        st.cache_data.clear()
    st.caption(f"Data through {max_d:%b %d, %Y}")

if not sites:
    st.warning("Pick at least one site.")
    st.stop()
start, end = (date_range if isinstance(date_range, tuple) and len(date_range) == 2 else (min_d, max_d))
site_list = ", ".join(f"'{s}'" for s in sites)

st.title("Warehouse Executive Monitor")
tab_overview, tab_score, tab_trends, tab_customers, tab_events = st.tabs(
    ["Overview", "Monthly scorecard", "Trends", "Customers", "Events"])

# ---------------------------------------------------------------- overview
with tab_overview:
    head = query(f"SELECT * FROM {T}.gold_exec_headline")
    net = head[head.warehouse_id == "NETWORK"].iloc[0]
    st.subheader(f"Network, last 7 days to {pd.to_datetime(net.as_of_date):%b %d}")

    def pts(a, b):
        return None if pd.isna(a) or pd.isna(b) else f"{a - b:+.1f} pts"

    c = st.columns(4)
    c[0].metric("Orders shipped", f"{net.orders_shipped_7d:,.0f}", f"{net.orders_change_pct:+.1f}%")
    c[1].metric("OTIF", f"{net.otif_pct_7d:.1f}%", pts(net.otif_pct_7d, net.otif_pct_prior_7d))
    c[2].metric("Labor cost / unit", f"${net.labor_cost_per_unit_7d:.2f}",
                f"{net.labor_cost_per_unit_7d - net.labor_cost_per_unit_prior_7d:+.3f}", delta_color="inverse")
    c[3].metric("Overtime", f"{net.overtime_pct_7d:.1f}%", pts(net.overtime_pct_7d, net.overtime_pct_prior_7d),
                delta_color="inverse")
    c = st.columns(4)
    c[0].metric("Open backlog", f"{net.open_backlog_orders:,.0f} orders")
    c[1].metric("Storage utilization", f"{net.utilization_pct:.1f}%")
    c[2].metric("Recordable injuries", f"{net.recordable_incidents_7d:.0f}")
    c[3].metric("Unplanned downtime", f"{net.unplanned_downtime_hours_7d:.1f} hrs")

    sh = head[head.warehouse_id.isin(sites)].copy()
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Sites at a glance**")
        st.dataframe(
            sh[["warehouse_name", "orders_shipped_7d", "otif_pct_7d", "fill_rate_pct_7d", "utilization_pct",
                "open_backlog_orders", "labor_cost_per_unit_7d", "overtime_pct_7d"]],
            hide_index=True, width="stretch",
            column_config={
                "warehouse_name": "Site",
                "orders_shipped_7d": st.column_config.NumberColumn("Orders (7d)", format="%d"),
                "otif_pct_7d": st.column_config.NumberColumn("OTIF %", format="%.1f"),
                "fill_rate_pct_7d": st.column_config.NumberColumn("Fill %", format="%.2f"),
                "utilization_pct": st.column_config.ProgressColumn("Utilization", format="%.0f%%",
                                                                   min_value=0, max_value=100),
                "open_backlog_orders": st.column_config.NumberColumn("Backlog", format="%d"),
                "labor_cost_per_unit_7d": st.column_config.NumberColumn("Labor $/unit", format="$%.2f"),
                "overtime_pct_7d": st.column_config.NumberColumn("OT %", format="%.1f"),
            })
    with right:
        fig = px.scatter_map(sh, lat="latitude", lon="longitude", size="orders_shipped_7d", color="otif_pct_7d",
                             hover_name="warehouse_name", color_continuous_scale="RdYlGn", range_color=(88, 99),
                             size_max=40, zoom=4.6, center={"lat": 37.6, "lon": -86.4}, map_style="carto-positron",
                             labels={"otif_pct_7d": "OTIF %", "orders_shipped_7d": "Orders"})
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=330)
        st.plotly_chart(fig, width="stretch")
        st.caption("Bubble size is orders shipped in the last 7 days; color is OTIF.")

# ---------------------------------------------------------------- scorecard
with tab_score:
    months = query(f"SELECT DISTINCT month_start FROM {T}.gold_monthly_scorecard ORDER BY month_start DESC")
    month = st.selectbox("Month", months.month_start, format_func=lambda m: f"{pd.to_datetime(m):%B %Y}")
    sc = query(f"SELECT * FROM {T}.gold_monthly_scorecard WHERE month_start = '{pd.to_datetime(month).date()}'")
    sc = sc[sc.warehouse_id.isin(sites + ["NETWORK"])]
    sc["label"] = sc.warehouse_id.map(lambda w: name_of.get(w, "Network Total"))
    order = sc.sort_values("display_order")[["kpi_name", "unit"]].drop_duplicates()

    def fmt(v, unit):
        if pd.isna(v):
            return "n/a"
        return {"%": f"{v:.1f}%", "$": f"${v:.2f}", "hours": f"{v:.1f}h"}.get(unit, f"{v:.1f}")

    sc["cell"] = [fmt(v, u) for v, u in zip(sc.actual_value, sc.unit)]
    cols = [name_of[s] for s in sites] + ["Network Total"]
    vals = sc.pivot(index="kpi_name", columns="label", values="cell").reindex(index=order.kpi_name, columns=cols)
    stat = sc.pivot(index="kpi_name", columns="label", values="status").reindex(index=order.kpi_name, columns=cols)
    styled = vals.style.apply(
        lambda _: stat.map(lambda s: f"background-color: {STATUS_COLORS.get(s, '#00000000')}; color: white"
                           if isinstance(s, str) else ""), axis=None)
    st.dataframe(styled, width="stretch", height=440)
    st.caption("Green meets target, Amber is within the warning band, Red is outside it. "
               "TRIR is a rolling 12-month rate; labor cost is scored against each site's own budget.")

    reds = sc[(sc.status == "Red") & (sc.warehouse_id != "NETWORK")].sort_values(["display_order", "label"])
    if not reds.empty:
        st.markdown("**Needs attention**")
        for r in reds.itertuples():
            chg = "" if pd.isna(r.change_vs_prior_month) else f" ({r.change_vs_prior_month:+.2f} vs prior month)"
            st.markdown(f"🔴 **{r.label}**: {r.kpi_name} at {fmt(r.actual_value, r.unit)} against a target of "
                        f"{fmt(r.target_value, r.unit)}{chg}")

# ---------------------------------------------------------------- trends
with tab_trends:
    kpi = st.selectbox("KPI", list(KPIS))
    num, den, scale, code, _, _ = KPIS[kpi]
    daily = query(f"""
        SELECT * FROM {T}.gold_site_daily_kpis
        WHERE kpi_date BETWEEN '{start}' AND '{end}' AND warehouse_id IN ({site_list})""")
    daily["kpi_date"] = pd.to_datetime(daily.kpi_date)
    daily = daily.sort_values(["warehouse_id", "kpi_date"])
    g = daily.groupby("warehouse_id")
    if den is None:
        daily["value"] = daily[num] if kpi.startswith("Open") else g[num].transform(lambda s: s.rolling(7, 1).sum())
    else:
        daily["value"] = scale * g[num].transform(lambda s: s.rolling(7, 1).sum()) / \
            g[den].transform(lambda s: s.rolling(7, 1).sum()).replace(0, pd.NA)
    daily["site"] = daily.warehouse_id.map(name_of)
    fig = px.line(daily, x="kpi_date", y="value", color="site",
                  labels={"kpi_date": "", "value": kpi, "site": "Site"})
    tgt = target_for(code) if code else None
    if tgt is not None and code != "LABOR_COST_PER_UNIT":
        fig.add_hline(y=tgt, line_dash="dash", line_color="gray", annotation_text="Target")
    ev = events[events.warehouse_id.isin(sites)].copy()
    ev["start_date"], ev["end_date"] = pd.to_datetime(ev.start_date), pd.to_datetime(ev.end_date)
    ev = ev[(ev.end_date >= pd.Timestamp(start)) & (ev.start_date <= pd.Timestamp(end))]
    for e in ev.drop_duplicates("event_id").itertuples():
        fig.add_vrect(x0=e.start_date, x1=e.end_date + pd.Timedelta(days=1), fillcolor="orange", opacity=0.12,
                      line_width=0, annotation_text=e.title, annotation_position="top left",
                      annotation_font_size=10)
    fig.update_layout(height=460, legend=dict(orientation="h", y=-0.15), margin=dict(t=30))
    st.plotly_chart(fig, width="stretch")
    st.caption("Lines are 7-day rolling values; shaded bands are operational events.")

# ---------------------------------------------------------------- customers
with tab_customers:
    cm = query(f"""
        SELECT * FROM {T}.gold_customer_monthly_service
        WHERE month_start = (SELECT MAX(month_start) FROM {T}.gold_customer_monthly_service)
          AND warehouse_id IN ({site_list})""")
    agg = (cm.groupby(["customer_name", "segment", "contract_tier", "otif_target_pct"], as_index=False)
             [["orders_shipped", "orders_otif", "fulfillment_revenue_usd"]].sum())
    agg["otif_pct"] = 100 * agg.orders_otif / agg.orders_shipped
    agg["gap"] = agg.otif_pct - agg.otif_target_pct
    agg = agg.sort_values("gap")
    st.subheader(f"Client service, {pd.to_datetime(cm.month_start.iloc[0]):%B %Y}")
    fig = go.Figure()
    fig.add_bar(y=agg.customer_name, x=agg.otif_pct, orientation="h", name="OTIF %",
                marker_color=["#d64545" if gp < 0 else "#2e9e5b" for gp in agg.gap])
    fig.add_scatter(y=agg.customer_name, x=agg.otif_target_pct, mode="markers", name="Target",
                    marker=dict(symbol="line-ns-open", size=18, color="gray", line_width=3))
    fig.update_layout(height=420, xaxis_range=[85, 100], margin=dict(t=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(agg[["customer_name", "segment", "contract_tier", "orders_shipped", "otif_pct", "otif_target_pct",
                      "fulfillment_revenue_usd"]], hide_index=True, width="stretch",
                 column_config={"customer_name": "Client", "orders_shipped": "Orders",
                                "otif_pct": st.column_config.NumberColumn("OTIF %", format="%.1f"),
                                "otif_target_pct": st.column_config.NumberColumn("Target %", format="%.1f"),
                                "fulfillment_revenue_usd": st.column_config.NumberColumn("Revenue", format="$%.0f")})

# ---------------------------------------------------------------- events
with tab_events:
    ev = events[events.warehouse_id.isin(sites)].copy()
    ev["site"] = ev.warehouse_id.map(name_of)
    st.dataframe(ev[["start_date", "end_date", "site", "event_type", "severity", "title", "description"]],
                 hide_index=True, width="stretch")
