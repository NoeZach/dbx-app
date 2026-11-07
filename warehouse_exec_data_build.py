# Databricks notebook source
# MAGIC %md
# MAGIC # Warehouse Executive Monitoring: Data Build
# MAGIC
# MAGIC This notebook generates twelve months (Nov 2025 through Oct 2026) of realistic, fully synthetic operations data for a five-site 3PL distribution network and lands it in Unity Catalog as a star schema, followed by a gold layer of executive KPI tables that an app or dashboard can query directly without any further transformation.
# MAGIC
# MAGIC | Layer | Tables |
# MAGIC |---|---|
# MAGIC | Dimensions | `dim_date`, `dim_warehouse`, `dim_customer`, `bridge_customer_warehouse`, `dim_operational_event`, `dim_kpi_target` |
# MAGIC | Facts | `fact_outbound_orders`, `fact_inbound_receipts`, `fact_labor_daily`, `fact_inventory_daily`, `fact_cycle_counts`, `fact_safety_incidents`, `fact_equipment_downtime` |
# MAGIC | Gold (app-ready) | `gold_site_daily_kpis`, `gold_network_daily_kpis`, `gold_monthly_scorecard`, `gold_customer_monthly_service`, `gold_exec_headline` |
# MAGIC
# MAGIC The data has storylines baked in so an executive app has something real to surface: holiday peak pressure in Nov and Dec, a winter storm in late January, a six hour WMS outage at Shepherdsville in March, a sorter failure at Columbus in May, a summer labor shortage at Memphis (OTIF, overtime and safety all degrade), a brand new Indianapolis site that ramps up and improves all year, and Columbus running hot on storage capacity going into peak. Every event is listed in `dim_operational_event` so the app can annotate charts.
# MAGIC
# MAGIC Run all cells. Set the catalog and schema with the widgets (the defaults work on Databricks Free Edition). The seed is fixed, so rerunning overwrites the tables with identical data. The notebook also runs outside Databricks, where it falls back to an in-memory DuckDB database for testing.

# COMMAND ----------

# MAGIC %md ## 1. Configuration

# COMMAND ----------

import numpy as np
import pandas as pd
from datetime import date

try:
    dbutils.widgets.text("catalog", "workspace", "Catalog")
    dbutils.widgets.text("schema", "warehouse_exec", "Schema")
    CATALOG = dbutils.widgets.get("catalog")
    SCHEMA = dbutils.widgets.get("schema")
    ON_DATABRICKS = True
except NameError:
    CATALOG, SCHEMA = "workspace", "warehouse_exec"
    ON_DATABRICKS = False

START_DATE = date(2025, 11, 1)
END_DATE = date(2026, 10, 31)
SEED = 42
rng = np.random.default_rng(SEED)
START_TS, END_TS = pd.Timestamp(START_DATE), pd.Timestamp(END_DATE)

# COMMAND ----------

# MAGIC %md ## 2. Reference data: calendar, warehouses, customers, events, KPI targets

# COMMAND ----------

HOLIDAYS = {
    "2025-11-27": "Thanksgiving",
    "2025-12-25": "Christmas Day",
    "2026-01-01": "New Year's Day",
    "2026-05-25": "Memorial Day",
    "2026-07-03": "Independence Day (observed)",
    "2026-09-07": "Labor Day",
}
HOL_TS = pd.to_datetime(list(HOLIDAYS))

dim_date = pd.DataFrame({"calendar_date": pd.date_range(START_DATE, END_DATE, freq="D")})
d = dim_date["calendar_date"]
dim_date.insert(0, "date_key", d.dt.strftime("%Y%m%d").astype(int))
dim_date["day_name"] = d.dt.day_name()
dim_date["day_of_week"] = d.dt.dayofweek + 1  # 1 = Monday
dim_date["is_weekend"] = dim_date["day_of_week"] >= 6
dim_date["week_start_date"] = d - pd.to_timedelta(d.dt.dayofweek, unit="D")
dim_date["month_start_date"] = d.dt.to_period("M").dt.to_timestamp()
dim_date["month_label"] = d.dt.strftime("%b %Y")
dim_date["quarter_label"] = "Q" + d.dt.quarter.astype(str) + " " + d.dt.year.astype(str)
dim_date["year"] = d.dt.year
hol = d.dt.strftime("%Y-%m-%d").map(HOLIDAYS)
dim_date["is_holiday"] = hol.notna()
dim_date["holiday_name"] = hol
dim_date["is_peak_season"] = (d.dt.month == 11) | ((d.dt.month == 12) & (d.dt.day <= 24))

dim_warehouse = pd.DataFrame(
    [
        ("WH01", "Shepherdsville DC", "Shepherdsville", "KY", "Ohio Valley", 37.9884, -85.7158, 1_100_000, 96, 62000, "2014-04-01", "Multi-client retail and e-commerce"),
        ("WH02", "Louisville Air Hub", "Louisville", "KY", "Ohio Valley", 38.1744, -85.7360, 420_000, 48, 18000, "2018-09-01", "Expedited and next-day healthcare"),
        ("WH03", "Memphis South DC", "Memphis", "TN", "Mid-South", 35.0424, -89.9767, 850_000, 80, 48000, "2016-03-01", "Retail replenishment and healthcare"),
        ("WH04", "Indianapolis East DC", "Indianapolis", "IN", "Great Lakes", 39.7790, -85.9990, 600_000, 60, 34000, "2025-09-15", "Industrial and e-commerce (new site)"),
        ("WH05", "Columbus Rickenbacker DC", "Columbus", "OH", "Great Lakes", 39.8137, -82.9279, 720_000, 72, 40000, "2019-06-01", "Food and beverage, retail"),
    ],
    columns=["warehouse_id", "warehouse_name", "city", "state", "region", "latitude", "longitude",
             "square_feet", "dock_doors", "pallet_positions", "opened_date", "operation_type"],
)
dim_warehouse["opened_date"] = pd.to_datetime(dim_warehouse["opened_date"])

# Site behaviour profiles (not written to tables; they drive the simulation)
PROFILE = {
    "WH01": dict(orders=420, late=0.022, short=0.010, acc=(0.986, 0.989), util=0.80, uph=118, wknd=(0.30, 0.10), trir=2.8, wage=21.75),
    "WH02": dict(orders=300, late=0.014, short=0.006, acc=(0.992, 0.994), util=0.72, uph=104, wknd=(0.65, 0.45), trir=2.4, wage=23.50),
    "WH03": dict(orders=350, late=0.026, short=0.012, acc=(0.980, 0.981), util=0.79, uph=112, wknd=(0.30, 0.10), trir=3.2, wage=19.75),
    "WH04": dict(orders=230, late=0.040, short=0.022, acc=(0.940, 0.988), util=0.45, uph=92, wknd=(0.25, 0.05), trir=4.2, wage=20.50),
    "WH05": dict(orders=280, late=0.024, short=0.009, acc=(0.983, 0.985), util=0.85, uph=115, wknd=(0.35, 0.10), trir=3.0, wage=21.00),
}

dim_customer = pd.DataFrame(
    [
        ("C001", "Bluegrass Outfitters", "Retail", "Gold", 2, 95.0),
        ("C002", "Northpoint Home Goods", "Retail", "Silver", 2, 94.0),
        ("C003", "Summit Health Supply", "Healthcare", "Platinum", 0, 98.5),
        ("C004", "RiverCity Pharma Logistics", "Healthcare", "Platinum", 0, 98.5),
        ("C005", "Copperline Industrial", "Industrial", "Gold", 3, 95.0),
        ("C006", "Hartwell Auto Parts", "Industrial", "Silver", 3, 94.0),
        ("C007", "Brightcart Online", "E-commerce", "Platinum", 1, 97.0),
        ("C008", "Maple & Pine Direct", "E-commerce", "Gold", 1, 96.0),
        ("C009", "Ohio Valley Beverages", "Food & Beverage", "Gold", 2, 96.0),
        ("C010", "Golden Fields Snacks", "Food & Beverage", "Silver", 2, 95.0),
        ("C011", "Trailhead Sporting Goods", "Retail", "Silver", 2, 94.0),
        ("C012", "Lumen Electronics", "E-commerce", "Gold", 1, 96.0),
    ],
    columns=["customer_id", "customer_name", "segment", "contract_tier", "sla_ship_days", "otif_target_pct"],
)

CUSTOMER_MIX = {
    "WH01": [("C001", .20), ("C002", .15), ("C007", .25), ("C008", .15), ("C011", .15), ("C012", .10)],
    "WH02": [("C003", .30), ("C004", .30), ("C007", .20), ("C012", .20)],
    "WH03": [("C001", .20), ("C002", .20), ("C003", .25), ("C004", .15), ("C011", .20)],
    "WH04": [("C005", .35), ("C006", .30), ("C008", .20), ("C012", .15)],
    "WH05": [("C009", .40), ("C010", .35), ("C002", .15), ("C011", .10)],
}
WH04_GOLIVE = pd.Timestamp("2026-04-01")  # Lumen Electronics launches at Indianapolis

bridge_customer_warehouse = pd.DataFrame(
    [(c, w, s) for w, mix in CUSTOMER_MIX.items() for c, s in mix],
    columns=["customer_id", "warehouse_id", "planned_volume_share"],
)
bridge_customer_warehouse["go_live_date"] = np.where(
    (bridge_customer_warehouse.customer_id == "C012") & (bridge_customer_warehouse.warehouse_id == "WH04"),
    WH04_GOLIVE, START_TS)
bridge_customer_warehouse["go_live_date"] = pd.to_datetime(bridge_customer_warehouse["go_live_date"])

EVENTS = [
    dict(event_id="EV001", sites=["ALL"], start="2025-11-28", end="2025-12-01", event_type="Demand spike",
         title="Black Friday / Cyber Monday surge", severity="Medium",
         description="Order volume ran roughly 40% above plan across the network over the Cyber 5 weekend.", vol=1.40),
    dict(event_id="EV002", sites=["WH01", "WH02", "WH04", "WH05"], start="2026-01-24", end="2026-01-26", event_type="Weather",
         title="Winter storm", severity="High",
         description="Ice and snow closed roads and grounded carrier linehauls, cutting outbound capacity sharply.", cap=0.40, late=0.22),
    dict(event_id="EV003", sites=["WH01"], start="2026-03-17", end="2026-03-17", event_type="System outage",
         title="WMS outage (6 hours)", severity="High",
         description="Warehouse management system was unavailable for six hours on first shift and orders backed up into the following days.",
         cap=0.50, late=0.15, downtime=("WMS / IT", 360, "Database failover failed")),
    dict(event_id="EV004", sites=["WH05"], start="2026-05-12", end="2026-05-12", event_type="Equipment failure",
         title="Sortation system failure", severity="Medium",
         description="Primary sorter drive motor failed, causing 14 hours of downtime while the site sorted manually.",
         cap=0.65, late=0.09, downtime=("Conveyor / Sorter", 840, "Drive motor failure")),
    dict(event_id="EV005", sites=["WH03"], start="2026-07-06", end="2026-09-30", event_type="Labor",
         title="Labor shortage", severity="High",
         description="A regional labor shortage left second shift understaffed, forcing heavy overtime and temp labor use.",
         cap=0.86, late=0.035, ot=0.10, temp=0.20, acc_drop=0.015, inc=1.8, uph_mult=0.87, d2s=1.45),
    dict(event_id="EV006", sites=["WH04"], start="2026-04-01", end="2026-04-01", event_type="Client go-live",
         title="Lumen Electronics go-live", severity="Low",
         description="New e-commerce client launched at Indianapolis East, adding roughly 15% volume."),
]
dim_operational_event = pd.DataFrame(
    [
        dict(event_id=e["event_id"], warehouse_id=w, start_date=pd.Timestamp(e["start"]), end_date=pd.Timestamp(e["end"]),
             event_type=e["event_type"], title=e["title"], severity=e["severity"], description=e["description"])
        for e in EVENTS
        for w in (list(PROFILE) if e["sites"] == ["ALL"] else e["sites"])
    ]
)

dim_kpi_target = pd.DataFrame(
    [
        ("OTIF", "On-Time In-Full", "%", 95.0, 92.0, "higher_is_better", 1, "Share of shipped orders that left on or before the promised date with every unit."),
        ("ON_TIME", "On-Time Ship", "%", 97.0, 94.0, "higher_is_better", 2, "Share of shipped orders that left on or before the promised date."),
        ("FILL_RATE", "Unit Fill Rate", "%", 99.5, 99.0, "higher_is_better", 3, "Units shipped divided by units ordered on shipped orders."),
        ("DOCK_TO_STOCK", "Dock-to-Stock", "hours", 10.0, 14.0, "lower_is_better", 4, "Average hours from trailer arrival to putaway complete."),
        ("UTILIZATION", "Storage Utilization", "%", 85.0, 92.0, "lower_is_better", 5, "Occupied pallet positions divided by capacity; above 92% is congestion risk."),
        ("INV_ACCURACY", "Inventory Accuracy", "%", 99.0, 98.0, "higher_is_better", 6, "Share of cycle counts with zero variance."),
        ("PICK_UPH", "Pick Productivity", "units/hr", 110.0, 100.0, "higher_is_better", 7, "Units picked per picking labor hour."),
        ("OVERTIME_PCT", "Overtime", "%", 8.0, 12.0, "lower_is_better", 8, "Overtime hours as a share of all labor hours."),
        ("LABOR_COST_PER_UNIT", "Labor Cost per Unit", "$", 0.0, 0.0, "lower_is_better", 9, "Total labor cost divided by units shipped (targets calibrated below)."),
        ("TRIR", "Total Recordable Incident Rate", "rate", 3.0, 4.0, "lower_is_better", 10, "Recordable incidents per 200,000 labor hours (OSHA basis)."),
        ("DOWNTIME_HOURS", "Unplanned Equipment Downtime", "hours", 25.0, 40.0, "lower_is_better", 11, "Unplanned equipment downtime hours in the period."),
    ],
    columns=["kpi_code", "kpi_name", "unit", "target_value", "warning_value", "direction", "display_order", "description"],
)
dim_kpi_target.insert(0, "warehouse_id", "ALL")  # 'ALL' = default; site rows override it
dim_kpi_target.loc[dim_kpi_target.kpi_code == "INV_ACCURACY", ["target_value", "warning_value"]] = [98.5, 97.5]
dim_kpi_target.loc[dim_kpi_target.kpi_code == "TRIR", "description"] = "Rolling 12-month recordable incidents per 200,000 labor hours (OSHA basis)."

# COMMAND ----------

# MAGIC %md ## 3. Daily site drivers (volume, stress, event effects)

# COMMAND ----------

site_day = dim_warehouse[["warehouse_id"]].merge(dim_date[["calendar_date", "day_of_week", "is_holiday", "is_peak_season"]], how="cross")
wh = site_day["warehouse_id"].values
cal = site_day["calendar_date"]
t = (cal - START_TS).dt.days.values
month = cal.dt.month.values
dow = site_day["day_of_week"].values
prof = lambda k: np.array([PROFILE[w][k] for w in wh])

WEEKDAY_F = np.array([1.15, 1.10, 1.05, 1.00, 0.95])
sat = np.array([PROFILE[w]["wknd"][0] for w in wh])
sun = np.array([PROFILE[w]["wknd"][1] for w in wh])
wf = np.where(dow <= 5, WEEKDAY_F[np.clip(dow - 1, 0, 4)], np.where(dow == 6, sat, sun))

MONTH_F = {11: 1.25, 12: 1.28, 1: 0.85, 2: 0.88, 3: 0.95, 4: 0.97, 5: 1.00, 6: 1.00, 7: 0.97, 8: 1.05, 9: 1.03, 10: 1.12}
mf = np.array([MONTH_F[m] for m in month])
mf = np.where((month == 12) & (cal.dt.day.values >= 21), 0.75, mf)

ramp = np.where(wh == "WH04", np.minimum(1.0, 0.55 + 0.45 * t / 180), 1.0)        # new site volume ramp
golive = np.where((wh == "WH04") & (cal.values >= WH04_GOLIVE.to_datetime64()), 1.15, 1.0)
trend = 1 + 0.06 * t / 365
holiday = np.where(site_day["is_holiday"].values, 0.15, 1.0)

for c, v in dict(vol=1.0, cap=1.0, late_add=0.0, ot_add=0.0, temp_add=0.0, acc_drop=0.0, inc_mult=1.0, uph_mult=1.0, d2s_mult=1.0).items():
    site_day[c] = v
for ev in EVENTS:
    sites = list(PROFILE) if ev["sites"] == ["ALL"] else ev["sites"]
    s, e = pd.Timestamp(ev["start"]), pd.Timestamp(ev["end"])
    m = site_day.warehouse_id.isin(sites) & cal.between(s, e)
    site_day.loc[m, "vol"] *= ev.get("vol", 1.0)
    site_day.loc[m, "cap"] *= ev.get("cap", 1.0)
    site_day.loc[m, "late_add"] += ev.get("late", 0.0)
    site_day.loc[m, "ot_add"] += ev.get("ot", 0.0)
    site_day.loc[m, "temp_add"] += ev.get("temp", 0.0)
    site_day.loc[m, "acc_drop"] += ev.get("acc_drop", 0.0)
    site_day.loc[m, "inc_mult"] *= ev.get("inc", 1.0)
    site_day.loc[m, "uph_mult"] *= ev.get("uph_mult", 1.0)
    site_day.loc[m, "d2s_mult"] *= ev.get("d2s", 1.0)
    if "late" in ev:  # recovery tail after a disruption
        for k, f in [(1, 0.5), (2, 0.25)]:
            mt = site_day.warehouse_id.isin(sites) & (cal == e + pd.Timedelta(days=k))
            site_day.loc[mt, "late_add"] += ev["late"] * f

normal = prof("orders") * wf * trend * ramp * golive
expected = normal * mf * site_day["vol"].values * holiday * rng.lognormal(0, 0.07, len(site_day))
site_day["n_orders"] = rng.poisson(expected)
site_day["stress"] = site_day["n_orders"] / (normal * 1.18 * site_day["cap"].values)
over = np.maximum(site_day["stress"].values - 1.0, 0)

late_ramp = np.where(wh == "WH04", 1.6 - 0.7 * np.minimum(1, t / 240), 1.0)
site_day["p_late"] = np.clip(prof("late") * late_ramp + site_day["late_add"].values + 0.30 * over, 0.003, 0.65)
short_ramp = np.where(wh == "WH04", 1.8 - 0.8 * np.minimum(1, t / 240), 1.0)
site_day["p_short"] = np.clip(prof("short") * short_ramp + 0.5 * site_day["acc_drop"].values + 0.02 * over, 0.001, 0.2)
site_day["t"], site_day["month"] = t, month

# COMMAND ----------

# MAGIC %md ## 4. Outbound orders

# COMMAND ----------

rep = np.repeat(np.arange(len(site_day)), site_day["n_orders"].values)
o = site_day.iloc[rep][["warehouse_id", "calendar_date", "p_late", "p_short", "stress"]].reset_index(drop=True)
N = len(o)
owh = o["warehouse_id"].values

cust = np.empty(N, dtype=object)
for w, mix in CUSTOMER_MIX.items():
    m = owh == w
    ids, wts = zip(*mix)
    cust[m] = rng.choice(ids, size=m.sum(), p=np.array(wts) / sum(wts))
pre_golive = (owh == "WH04") & (cust == "C012") & (o["calendar_date"].values < WH04_GOLIVE.to_datetime64())
cust[pre_golive] = "C008"
o["customer_id"] = cust
o = o.merge(dim_customer[["customer_id", "segment", "contract_tier", "sla_ship_days"]], on="customer_id", how="left")
seg = o["segment"].values

hours = np.clip(rng.normal(11.5, 3.2, N), 0, 23.98)
o["order_ts"] = o["calendar_date"] + pd.to_timedelta(np.round(hours * 60), unit="m")


def roll_forward(dates, wh_arr, skip_saturday):
    """Push dates off closed days: holidays and Sundays for every site except WH02, plus Saturdays when requested."""
    dts = dates.copy()
    for _ in range(5):
        dw = dts.dt.dayofweek.values
        bad = dts.isin(HOL_TS).values | ((dw == 6) & (wh_arr != "WH02"))
        if skip_saturday:
            bad |= (dw == 5) & (wh_arr != "WH02")
        if not bad.any():
            break
        dts = dts + pd.to_timedelta(bad.astype(int), unit="D")
    return dts


sla = o["sla_ship_days"].values + ((o["sla_ship_days"].values == 0) & (hours >= 15)).astype(int)  # same-day cutoff 3pm
promised = roll_forward(o["calendar_date"] + pd.to_timedelta(sla, unit="D"), owh, skip_saturday=True)

is_late = rng.random(N) < o["p_late"].values
days_late = 1 + rng.poisson(0.3 + 1.5 * np.maximum(o["stress"].values - 1, 0))
early = (~is_late) & (sla >= 1) & (rng.random(N) < 0.30)
ship = promised + pd.to_timedelta(np.where(is_late, days_late, 0) - early.astype(int), unit="D")
ship = ship.where(ship >= o["calendar_date"], o["calendar_date"])
ship = roll_forward(ship, owh, skip_saturday=False)

LINES = {"Retail": 6, "E-commerce": 1.6, "Healthcare": 3, "Industrial": 4, "Food & Beverage": 8}
UPL = {"Retail": 12, "E-commerce": 1.4, "Healthcare": 6, "Industrial": 9, "Food & Beverage": 22}
lines = 1 + rng.poisson(np.array([LINES[s] for s in seg]) - 1)
upl_mean = np.array([UPL[s] for s in seg])
units = np.maximum(lines, np.round(lines * upl_mean * rng.lognormal(-0.18, 0.6, N))).astype(int)

is_short = rng.random(N) < o["p_short"].values
short_units = np.minimum(units, np.maximum(1, np.round(units * rng.uniform(0.02, 0.25, N)))).astype(int)
units_shipped = np.where(is_short, units - short_units, units)

METHODS = {
    "Healthcare": (["Next Day", "2-Day", "Ground"], [.5, .3, .2]),
    "E-commerce": (["Ground", "2-Day", "Next Day"], [.7, .25, .05]),
    "Retail": (["LTL", "Ground", "Truckload"], [.6, .25, .15]),
    "Industrial": (["LTL", "Truckload", "Ground"], [.55, .3, .15]),
    "Food & Beverage": (["Truckload", "LTL", "Ground"], [.45, .5, .05]),
}
method = np.empty(N, dtype=object)
for s_, (ms, ps) in METHODS.items():
    m = seg == s_
    method[m] = rng.choice(ms, size=m.sum(), p=ps)
CARRIERS = {
    "Ground": (["UPS", "FedEx", "USPS"], [.5, .4, .1]),
    "2-Day": (["UPS", "FedEx"], [.55, .45]),
    "Next Day": (["UPS", "FedEx"], [.5, .5]),
    "LTL": (["XPO", "Estes", "Old Dominion", "R+L Carriers"], [.3, .25, .25, .2]),
    "Truckload": (["Dedicated Fleet", "J.B. Hunt", "Schneider"], [.4, .3, .3]),
}
carrier = np.empty(N, dtype=object)
for mth, (cs, ps) in CARRIERS.items():
    m = method == mth
    carrier[m] = rng.choice(cs, size=m.sum(), p=ps)

noise = rng.lognormal(0, 0.15, N)
parcel_base = np.select([method == "Ground", method == "2-Day", method == "Next Day"], [8.5, 16.0, 28.0], 0)
freight = np.select(
    [np.isin(method, ["Ground", "2-Day", "Next Day"]), method == "LTL", method == "Truckload"],
    [parcel_base * (1 + units / 12), 165 + 1.1 * units ** 0.85, np.full(N, 1100.0)], 0) * noise
TIER_FEE = {"Platinum": 3.25, "Gold": 3.75, "Silver": 4.25}
revenue = np.array([TIER_FEE[x] for x in o["contract_tier"].values]) + 0.38 * units + 0.25 * lines

is_open = (ship > END_TS).values
on_time = (ship <= promised).values

fact_outbound_orders = pd.DataFrame({
    "warehouse_id": owh,
    "customer_id": cust,
    "order_date": o["calendar_date"],
    "order_ts": o["order_ts"],
    "promised_ship_date": promised,
    "ship_date": ship.where(~is_open),
    "order_status": np.where(is_open, "Open", "Shipped"),
    "ship_method": method,
    "carrier": carrier,
    "order_lines": lines,
    "units_ordered": units,
    "units_shipped": pd.array(np.where(is_open, 0, units_shipped), dtype="Int64"),
    "is_on_time": pd.array(on_time, dtype="boolean"),
    "is_in_full": pd.array(~is_short, dtype="boolean"),
    "is_otif": pd.array(on_time & ~is_short, dtype="boolean"),
    "days_late": pd.array(np.maximum((ship - promised).dt.days.values, 0), dtype="Int64"),
    "freight_cost_usd": np.round(freight, 2),
    "fulfillment_revenue_usd": np.round(revenue, 2),
})
for c in ["units_shipped", "is_on_time", "is_in_full", "is_otif", "days_late"]:
    fact_outbound_orders.loc[is_open, c] = pd.NA
fact_outbound_orders.loc[is_open, ["freight_cost_usd"]] = 0.0
fact_outbound_orders = fact_outbound_orders.sort_values("order_ts", kind="stable").reset_index(drop=True)
fact_outbound_orders.insert(0, "order_id", "SO" + (fact_outbound_orders.index + 1_000_001).astype(str))

# COMMAND ----------

# MAGIC %md ## 5. Inbound receipts

# COMMAND ----------

BUILD = {9: 1.25, 10: 1.35, 11: 1.10, 12: 0.80, 1: 0.85}
build = np.array([BUILD.get(m, 1.0) for m in month])
in_wf = np.where(dow <= 5, 1.0, np.where(dow == 6, np.where(wh == "WH02", 0.4, 0.15), np.where(wh == "WH02", 0.2, 0.0)))
exp_in = prof("orders") * 0.085 * trend * ramp * golive * build * in_wf * holiday * (site_day["is_holiday"].values == False)
site_day["n_receipts"] = rng.poisson(exp_in * rng.lognormal(0, 0.1, len(site_day)))

rep = np.repeat(np.arange(len(site_day)), site_day["n_receipts"].values)
r = site_day.iloc[rep][["warehouse_id", "calendar_date", "stress", "t", "d2s_mult"]].reset_index(drop=True)
R = len(r)
rwh = r["warehouse_id"].values
rc = np.empty(R, dtype=object)
for w, mix in CUSTOMER_MIX.items():
    m = rwh == w
    ids, wts = zip(*mix)
    rc[m] = rng.choice(ids, size=m.sum(), p=np.array(wts) / sum(wts))
rc[(rwh == "WH04") & (rc == "C012") & (r["calendar_date"].values < WH04_GOLIVE.to_datetime64())] = "C008"
rseg = dim_customer.set_index("customer_id").loc[rc, "segment"].values

PALLETS = {"Retail": 22, "E-commerce": 20, "Healthcare": 10, "Industrial": 18, "Food & Beverage": 26}
UPP = {"Retail": 180, "E-commerce": 240, "Healthcare": 300, "Industrial": 90, "Food & Beverage": 72}
pallets = 1 + rng.poisson(np.array([PALLETS[s] for s in rseg]) - 1)
units_rcv = np.round(pallets * np.array([UPP[s] for s in rseg]) * rng.lognormal(-0.05, 0.3, R)).astype(int)

appt = r["calendar_date"] + pd.to_timedelta(rng.integers(10, 42, R) * 30, unit="m")  # 05:00 to 20:30
mins = np.round(rng.normal(8, 35, R) + np.where(rng.random(R) < 0.05, rng.uniform(120, 300, R), 0)).astype(int)
arrival = appt + pd.to_timedelta(mins, unit="m")

site_d2s = np.where(rwh == "WH04", 1.6 - 0.6 * np.minimum(1, r["t"].values / 240), 1.0)
d2s = 7.5 * rng.lognormal(0, 0.45, R) * site_d2s * r["d2s_mult"].values * (1 + 0.9 * np.maximum(r["stress"].values - 1, 0))

TRAILER = {"Food & Beverage": ([.45, .5, .05, 0], ), "Healthcare": ([.75, .2, .05, 0],), "Industrial": ([.65, .15, .05, .15],)}
trailer = np.empty(R, dtype=object)
for s_ in np.unique(rseg):
    m = rseg == s_
    p = TRAILER.get(s_, ([.75, 0, .22, .03],))[0]
    trailer[m] = rng.choice(["Dry Van", "Reefer", "Container", "Flatbed"], size=m.sum(), p=p)
asn_p = np.where(rwh == "WH04", 0.88 + 0.07 * np.minimum(1, r["t"].values / 240), 0.96)

fact_inbound_receipts = pd.DataFrame({
    "warehouse_id": rwh,
    "customer_id": rc,
    "receipt_date": r["calendar_date"],
    "appointment_ts": appt,
    "arrival_ts": arrival,
    "minutes_from_appointment": mins,
    "trailer_type": trailer,
    "pallets_received": pallets,
    "units_received": units_rcv,
    "units_damaged": rng.binomial(units_rcv, 0.0015),
    "asn_accurate": rng.random(R) < asn_p,
    "dock_to_stock_hours": np.round(d2s, 2),
    "putaway_complete_ts": arrival + pd.to_timedelta(np.round(d2s * 60), unit="m"),
}).sort_values("arrival_ts", kind="stable").reset_index(drop=True)
fact_inbound_receipts.insert(0, "receipt_id", "RC" + (fact_inbound_receipts.index + 500_001).astype(str))

# COMMAND ----------

# MAGIC %md ## 6. Labor by site, day, shift and function

# COMMAND ----------

shipped = fact_outbound_orders[fact_outbound_orders.order_status == "Shipped"]
ship_agg = (shipped.groupby(["warehouse_id", "ship_date"])
            .agg(orders_shipped=("order_id", "size"), units_shipped=("units_shipped", "sum"))
            .reset_index().rename(columns={"ship_date": "calendar_date"}))
in_agg = (fact_inbound_receipts.groupby(["warehouse_id", "receipt_date"])
          .agg(units_received=("units_received", "sum"), pallets_received=("pallets_received", "sum"))
          .reset_index().rename(columns={"receipt_date": "calendar_date"}))
wl = (site_day.merge(ship_agg, on=["warehouse_id", "calendar_date"], how="left")
      .merge(in_agg, on=["warehouse_id", "calendar_date"], how="left"))
for c in ["orders_shipped", "units_shipped", "units_received", "pallets_received"]:
    wl[c] = wl[c].fillna(0).astype(float)
wl = wl.merge(dim_warehouse[["warehouse_id", "pallet_positions"]], on="warehouse_id")
wwh, wt = wl["warehouse_id"].values, wl["t"].values
peak = wl["is_peak_season"].values.astype(float)
over = np.maximum(wl["stress"].values - 1, 0)

uph_ramp = np.where(wwh == "WH04", 0.85 + 0.25 * np.minimum(1, wt / 270), 1.0)
pick_rate = np.array([PROFILE[w]["uph"] for w in wwh]) * uph_ramp * wl["uph_mult"].values * (1 - 0.04 * peak)
open_day = (wl["day_of_week"].values <= 5) & ~wl["is_holiday"].values

FUNCS = [
    ("Receiving", wl["units_received"].values / 520),
    ("Putaway", wl["pallets_received"].values / 20),
    ("Picking", wl["units_shipped"].values / pick_rate),
    ("Packing", wl["units_shipped"].values / 160),
    ("Shipping", wl["orders_shipped"].values / 48),
    ("Inventory Control", np.where(open_day, 24 * wl["pallet_positions"].values / 40000, 0)),
]
_direct = sum(h for _, h in FUNCS)
FUNCS.append(("Indirect / Support", _direct * 0.35 + np.where(open_day | (wl["n_orders"].values > 0), 40 * wl["pallet_positions"].values / 40000, 0)))
ot_pct = np.clip(0.035 + 0.45 * over + wl["ot_add"].values + 0.03 * peak + rng.normal(0, 0.01, len(wl)), 0, 0.4)
temp_pct = np.clip(0.12 + 0.18 * peak + wl["temp_add"].values + np.where(wwh == "WH04", 0.12 * (1 - np.minimum(1, wt / 270)), 0), 0, 0.6)
wage = np.array([PROFILE[w]["wage"] for w in wwh])
units_by_func = {"Receiving": wl["units_received"].values, "Putaway": wl["pallets_received"].values,
                 "Picking": wl["units_shipped"].values, "Packing": wl["units_shipped"].values,
                 "Shipping": wl["orders_shipped"].values, "Inventory Control": np.zeros(len(wl)),
                 "Indirect / Support": np.zeros(len(wl))}

labor_parts = []
for shift, share in [("1st", 0.58), ("2nd", 0.42)]:
    extra_ot = np.where((shift == "2nd") & (wl["ot_add"].values > 0), 0.05, 0)
    for fn, base_hours in FUNCS:
        hrs = base_hours * share * rng.normal(1, 0.04, len(wl))
        o_pct = np.clip(ot_pct + extra_ot, 0, 0.45)
        hc = np.ceil(hrs / 8.25)
        cost = (hrs * (1 - o_pct) + hrs * o_pct * 1.5) * wage * (temp_pct * 1.35 + (1 - temp_pct) * 1.28)
        labor_parts.append(pd.DataFrame({
            "warehouse_id": wwh, "work_date": wl["calendar_date"].values, "shift": shift, "labor_function": fn,
            "headcount": hc.astype(int), "temp_headcount": np.round(hc * temp_pct).astype(int),
            "hours_worked": np.round(hrs, 1), "overtime_hours": np.round(hrs * o_pct, 1),
            "units_processed": np.round(units_by_func[fn] * share).astype(int),
            "labor_cost_usd": np.round(cost, 2),
        }))
fact_labor_daily = pd.concat(labor_parts, ignore_index=True)
fact_labor_daily = fact_labor_daily[fact_labor_daily["hours_worked"] >= 0.5].sort_values(["work_date", "warehouse_id", "shift", "labor_function"]).reset_index(drop=True)

# COMMAND ----------

# MAGIC %md ## 7. Inventory snapshots and cycle counts

# COMMAND ----------

ZONES = {
    "WH01": [("Ambient Rack", .55), ("Bulk Floor", .25), ("Pick Mezzanine", .15), ("Secure Cage", .05)],
    "WH02": [("Ambient Rack", .50), ("Cold Storage", .20), ("Pick Mezzanine", .20), ("Secure Cage", .10)],
    "WH03": [("Ambient Rack", .55), ("Bulk Floor", .20), ("Cold Storage", .15), ("Pick Mezzanine", .10)],
    "WH04": [("Ambient Rack", .60), ("Bulk Floor", .30), ("Pick Mezzanine", .10)],
    "WH05": [("Ambient Rack", .45), ("Bulk Floor", .30), ("Cold Storage", .25)],
}
ZONE_OFFSET = {"Ambient Rack": 0.0, "Bulk Floor": -0.03, "Pick Mezzanine": 0.03, "Cold Storage": 0.05, "Secure Cage": -0.12}
ZONE_VALUE = {"Ambient Rack": 1800, "Bulk Floor": 1200, "Pick Mezzanine": 2600, "Cold Storage": 2200, "Secure Cage": 14000}
ZONE_CODE = {"Ambient Rack": "AR", "Bulk Floor": "BF", "Pick Mezzanine": "PM", "Cold Storage": "CS", "Secure Cage": "SC"}
SEASON = {1: -.05, 2: -.04, 3: -.02, 4: 0, 5: 0, 6: .01, 7: .02, 8: .03, 9: .05, 10: .08, 11: .06, 12: .02}
WH05_CRUNCH = {10: .05, 11: .07, 12: .04}

dates = dim_date["calendar_date"]
season = pd.Series([SEASON[m] for m in dates.dt.month]).rolling(31, center=True, min_periods=1).mean().values
crunch = pd.Series([WH05_CRUNCH.get(m, 0) for m in dates.dt.month]).rolling(31, center=True, min_periods=1).mean().values
days = np.arange(len(dates))
inv_parts = []
for w, p in PROFILE.items():
    cap_total = int(dim_warehouse.set_index("warehouse_id").loc[w, "pallet_positions"])
    base = np.where(w == "WH04", 0.45 + 0.29 * np.minimum(1, days / 270), p["util"])
    ar = np.zeros(len(days))
    for i in range(1, len(days)):
        ar[i] = 0.9 * ar[i - 1] + rng.normal(0, 0.006)
    util = base + season + (crunch if w == "WH05" else 0) + ar
    aged = 0.08 + np.where(dates.dt.month.isin([1, 2]), 0.04, 0)
    for z, share in ZONES[w]:
        zu = np.clip(util + ZONE_OFFSET[z] + rng.normal(0, 0.008, len(days)), 0.2, 0.99)
        cap_z = int(round(cap_total * share))
        occ = np.round(cap_z * zu).astype(int)
        inv_parts.append(pd.DataFrame({
            "warehouse_id": w, "snapshot_date": dates.values, "zone": z, "pallet_capacity": cap_z,
            "pallets_occupied": occ, "utilization_pct": np.round(100 * occ / cap_z, 2),
            "inventory_value_usd": np.round(occ * ZONE_VALUE[z] * rng.lognormal(0, 0.03, len(days)), 2),
            "sku_count": np.round(occ * rng.uniform(0.55, 0.65, len(days))).astype(int),
            "aged_90_plus_pallets": np.round(occ * (aged + rng.normal(0, 0.005, len(days)))).astype(int),
        }))
fact_inventory_daily = pd.concat(inv_parts, ignore_index=True)

# Cycle counts
cc_days = site_day[(site_day["day_of_week"] <= 5) & ~site_day["is_holiday"]].merge(dim_warehouse[["warehouse_id", "pallet_positions"]], on="warehouse_id")
cc_days["n"] = rng.poisson(55 * cc_days["pallet_positions"] / 40000)
rep = np.repeat(np.arange(len(cc_days)), cc_days["n"].values)
c = cc_days.iloc[rep][["warehouse_id", "calendar_date", "t", "acc_drop", "is_peak_season"]].reset_index(drop=True)
C = len(c)
cwh = c["warehouse_id"].values
acc0 = np.array([PROFILE[w]["acc"][0] for w in cwh])
acc1 = np.array([PROFILE[w]["acc"][1] for w in cwh])
p_acc = np.clip(acc0 + (acc1 - acc0) * np.minimum(1, c["t"].values / 300) - c["acc_drop"].values - 0.004 * c["is_peak_season"].values, 0.85, 0.999)
accurate = rng.random(C) < p_acc
zone = np.empty(C, dtype=object)
for w, zs in ZONES.items():
    m = cwh == w
    names, shares = zip(*zs)
    zone[m] = rng.choice(names, size=m.sum(), p=np.array(shares) / sum(shares))
loc = (pd.Series([ZONE_CODE[z] for z in zone]) + "-" + pd.Series(rng.integers(1, 41, C)).astype(str).str.zfill(2)
       + "-" + pd.Series(rng.integers(1, 121, C)).astype(str).str.zfill(3) + "-" + pd.Series(rng.choice(list("ABCDE"), C)))
sys_qty = np.maximum(0, np.round(rng.lognormal(3.6, 0.9, C))).astype(int)
var = np.where(accurate, 0, rng.choice([-1, 1], C, p=[.62, .38]) * (1 + rng.geometric(0.35, C) - 1))
counted = np.maximum(0, sys_qty + var)
unit_cost = np.round(np.where(zone == "Secure Cage", 180, 18) * rng.lognormal(0, 0.5, C), 2)
fact_cycle_counts = pd.DataFrame({
    "warehouse_id": cwh, "count_date": c["calendar_date"], "zone": zone, "location_id": loc.values,
    "sku_id": "SKU-" + pd.Series(rng.integers(1, 60001, C)).astype(str).str.zfill(6),
    "count_type": rng.choice(["ABC Cycle", "Exception", "Blind"], C, p=[.75, .15, .10]),
    "system_qty": sys_qty, "counted_qty": counted, "variance_units": counted - sys_qty,
    "unit_cost_usd": unit_cost, "variance_value_usd": np.round((counted - sys_qty) * unit_cost, 2),
    "is_accurate": counted == sys_qty,
})
fact_cycle_counts.insert(0, "count_id", "CC" + (fact_cycle_counts.index + 2_000_001).astype(str))

# COMMAND ----------

# MAGIC %md ## 8. Safety incidents and equipment downtime

# COMMAND ----------

hours_day = fact_labor_daily.groupby(["warehouse_id", "work_date"])["hours_worked"].sum().rename("hours").reset_index()
sd = site_day.merge(hours_day, left_on=["warehouse_id", "calendar_date"], right_on=["warehouse_id", "work_date"], how="left").fillna({"hours": 0})
trir = np.array([PROFILE[w]["trir"] for w in sd["warehouse_id"]])
trir = np.where(sd["warehouse_id"] == "WH04", trir - 1.2 * np.minimum(1, sd["t"].values / 270), trir)
TYPES = {"Near Miss": 8.0, "First Aid": 2.5, "Recordable": 0.8, "Lost Time": 0.2, "Property Damage": 1.2}
AREAS = (["Receiving", "Putaway", "Picking", "Packing", "Shipping", "Dock Yard"], [.18, .14, .30, .14, .16, .08])
CAUSES = (["Powered industrial truck", "Slip, trip or fall", "Lifting strain", "Pinch point", "Struck by object", "Cut or laceration"],
          [.24, .20, .22, .10, .14, .10])
inc_parts = []
for typ, mult in TYPES.items():
    lam = trir * mult * sd["inc_mult"].values * sd["hours"].values / 200_000
    if typ in ("Recordable", "Lost Time"):
        # Rare events: follow each site's cumulative expected count so a single unlucky draw does not define a year,
        # then place each month's injuries on days weighted by that day's exposure.
        n = np.zeros(len(sd), dtype=int)
        for w in PROFILE:
            idx = np.where(sd["warehouse_id"].values == w)[0]
            cum = np.floor(np.cumsum(lam[idx]) + 0.5)
            hits = np.diff(np.concatenate([[0], cum])).astype(int)
            mo = sd["calendar_date"].values[idx].astype("datetime64[M]")
            for m_ in np.unique(mo):
                sel = idx[mo == m_]
                k_ = int(hits[mo == m_].sum())
                if k_:
                    np.add.at(n, rng.choice(sel, size=k_, p=lam[sel] / lam[sel].sum()), 1)
    else:
        n = rng.poisson(lam)
    rep = np.repeat(np.arange(len(sd)), n)
    if len(rep) == 0:
        continue
    sub = sd.iloc[rep]
    k = len(rep)
    inc_parts.append(pd.DataFrame({
        "warehouse_id": sub["warehouse_id"].values, "incident_date": sub["calendar_date"].values, "incident_type": typ,
        "is_recordable": typ in ("Recordable", "Lost Time"), "shift": rng.choice(["1st", "2nd"], k, p=[.52, .48]),
        "area": rng.choice(AREAS[0], k, p=AREAS[1]), "cause": rng.choice(CAUSES[0], k, p=CAUSES[1]),
        "lost_days": (rng.geometric(0.15, k) if typ == "Lost Time" else np.zeros(k, dtype=int)),
    }))
fact_safety_incidents = pd.concat(inc_parts, ignore_index=True).sort_values(["incident_date", "warehouse_id"]).reset_index(drop=True)
fact_safety_incidents.insert(0, "incident_id", "SI" + (fact_safety_incidents.index + 10_001).astype(str))

EQUIP = (["Conveyor / Sorter", "Forklift / PIT", "Dock Door / Leveler", "RF Scanner / Network", "WMS / IT", "Stretch Wrapper"], [.30, .30, .18, .10, .05, .07])
EQ_CODE = {"Conveyor / Sorter": "CNV", "Forklift / PIT": "FLT", "Dock Door / Leveler": "DCK", "RF Scanner / Network": "NET", "WMS / IT": "WMS", "Stretch Wrapper": "WRP"}
EQ_CAUSE = {"Conveyor / Sorter": ["Belt jam", "Photo-eye fault", "Motor overload"], "Forklift / PIT": ["Battery failure", "Hydraulic leak", "Mast damage"],
            "Dock Door / Leveler": ["Hydraulic failure", "Door panel damage", "Seal damage"], "RF Scanner / Network": ["Access point outage", "Device failure"],
            "WMS / IT": ["Application error", "Interface delay"], "Stretch Wrapper": ["Film carriage fault", "Turntable fault"]}
op = site_day[~site_day["is_holiday"] & (site_day["n_orders"] > 0)]
n = rng.poisson(0.55 * np.where(op["day_of_week"] >= 6, 0.3, 1.0) * (1 + 0.5 * op["is_peak_season"].values))
rep = np.repeat(np.arange(len(op)), n)
sub = op.iloc[rep]
k = len(rep)
eq = rng.choice(EQUIP[0], k, p=EQUIP[1])
start = sub["calendar_date"].values + pd.to_timedelta(rng.integers(300, 1380, k), unit="m").values
dmin = np.round(rng.lognormal(np.log(45), 0.8, k)).astype(int)
fact_equipment_downtime = pd.DataFrame({
    "warehouse_id": sub["warehouse_id"].values, "downtime_date": sub["calendar_date"].values, "start_ts": start,
    "equipment_type": eq, "asset_id": [f"{w}-{EQ_CODE[e]}-{i:02d}" for w, e, i in zip(sub["warehouse_id"].values, eq, rng.integers(1, 13, k))],
    "cause": [rng.choice(EQ_CAUSE[e]) for e in eq], "downtime_minutes": dmin,
    "is_unplanned": rng.random(k) < 0.9, "linked_event_id": None,
})
ev_rows = [dict(warehouse_id=ev["sites"][0], downtime_date=pd.Timestamp(ev["start"]), start_ts=pd.Timestamp(ev["start"]) + pd.Timedelta(hours=6),
                equipment_type=ev["downtime"][0], asset_id=f"{ev['sites'][0]}-{EQ_CODE[ev['downtime'][0]]}-01", cause=ev["downtime"][2],
                downtime_minutes=ev["downtime"][1], is_unplanned=True, linked_event_id=ev["event_id"]) for ev in EVENTS if "downtime" in ev]
fact_equipment_downtime = pd.concat([fact_equipment_downtime, pd.DataFrame(ev_rows)], ignore_index=True)
fact_equipment_downtime["end_ts"] = fact_equipment_downtime["start_ts"] + pd.to_timedelta(fact_equipment_downtime["downtime_minutes"], unit="m")
fact_equipment_downtime = fact_equipment_downtime.sort_values("start_ts").reset_index(drop=True)
fact_equipment_downtime.insert(0, "downtime_id", "DT" + (fact_equipment_downtime.index + 30_001).astype(str))

# Labor cost per unit depends heavily on each site's order profile (small healthcare parcels vs full pallets of beverages),
# so the default row covers the network and each site gets its own budget row, calibrated from its median month.
def _cpu(lab, shp):
    return (lab.groupby(lab["work_date"].dt.to_period("M"))["labor_cost_usd"].sum()
            / shp.groupby(shp["ship_date"].dt.to_period("M"))["units_shipped"].sum().astype(float))
_lc = dim_kpi_target.kpi_code == "LABOR_COST_PER_UNIT"
_med = float(np.median(_cpu(fact_labor_daily, shipped)))
dim_kpi_target.loc[_lc, ["target_value", "warning_value"]] = [round(_med * 0.97, 3), round(_med * 1.08, 3)]
dim_kpi_target.loc[_lc, "description"] = "Total labor cost divided by units shipped, against each site's budgeted rate."
_site_rows = []
for w in PROFILE:
    _med_w = float(np.median(_cpu(fact_labor_daily[fact_labor_daily.warehouse_id == w], shipped[shipped.warehouse_id == w])))
    r_ = dim_kpi_target[_lc].iloc[0].copy()
    r_["warehouse_id"], r_["target_value"], r_["warning_value"] = w, round(_med_w * 0.97, 3), round(_med_w * 1.08, 3)
    _site_rows.append(r_)
dim_kpi_target = pd.concat([dim_kpi_target, pd.DataFrame(_site_rows)], ignore_index=True)

print(f"orders={len(fact_outbound_orders):,}  receipts={len(fact_inbound_receipts):,}  labor_rows={len(fact_labor_daily):,}  "
      f"inventory_rows={len(fact_inventory_daily):,}  cycle_counts={len(fact_cycle_counts):,}  incidents={len(fact_safety_incidents):,}  "
      f"downtime_events={len(fact_equipment_downtime):,}")

# COMMAND ----------

# MAGIC %md ## 9. Write dimension and fact tables

# COMMAND ----------

TABLES = {
    "dim_date": (dim_date, ["calendar_date", "week_start_date", "month_start_date"],
                 "Calendar dimension with holidays and peak-season flags, Nov 2025 to Oct 2026."),
    "dim_warehouse": (dim_warehouse, ["opened_date"], "One row per distribution center with location, size and capacity."),
    "dim_customer": (dim_customer, [], "3PL clients with segment, contract tier, ship SLA in days and OTIF target."),
    "bridge_customer_warehouse": (bridge_customer_warehouse, ["go_live_date"], "Which clients run out of which sites, with planned volume share."),
    "dim_operational_event": (dim_operational_event, ["start_date", "end_date"], "Notable disruptions and milestones, one row per affected site, for chart annotations."),
    "dim_kpi_target": (dim_kpi_target, [], "Executive KPI targets and warning thresholds for red/amber/green status; warehouse_id ALL is the default and site rows override it."),
    "fact_outbound_orders": (fact_outbound_orders, ["order_date", "promised_ship_date", "ship_date"],
                             "One row per outbound customer order. OTIF flags are null while an order is still open."),
    "fact_inbound_receipts": (fact_inbound_receipts, ["receipt_date"], "One row per inbound trailer receipt with dock-to-stock time."),
    "fact_labor_daily": (fact_labor_daily, ["work_date"], "Labor hours, overtime, headcount and cost by site, day, shift and function."),
    "fact_inventory_daily": (fact_inventory_daily, ["snapshot_date"], "End-of-day storage utilization and inventory value by site and zone."),
    "fact_cycle_counts": (fact_cycle_counts, ["count_date"], "Individual cycle count results; is_accurate means zero variance."),
    "fact_safety_incidents": (fact_safety_incidents, ["incident_date"], "Safety events from near misses through lost-time injuries."),
    "fact_equipment_downtime": (fact_equipment_downtime, ["downtime_date"], "Equipment and system downtime events with duration and cause."),
}

if ON_DATABRICKS:
    from pyspark.sql import functions as F

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
    spark.sql(f"USE CATALOG {CATALOG}")
    spark.sql(f"USE SCHEMA {SCHEMA}")

    def to_spark(pdf, date_cols):
        pdf = pdf.copy()
        casts = {}
        for col, dt in pdf.dtypes.items():
            if str(dt) == "Int64":
                pdf[col], casts[col] = pdf[col].astype("float64"), "bigint"
            elif str(dt) == "boolean":
                pdf[col], casts[col] = pdf[col].astype("float64"), "boolean"
        sdf = spark.createDataFrame(pdf)
        for col, typ in dict(sdf.dtypes).items():
            if typ == "double":
                sdf = sdf.withColumn(col, F.when(F.isnan(col), F.lit(None)).otherwise(F.col(col)))
        for col, typ in casts.items():
            sdf = sdf.withColumn(col, F.col(col).cast(typ))
        for col in date_cols:
            sdf = sdf.withColumn(col, F.to_date(col))
        return sdf

    for name, (pdf, dcols, comment) in TABLES.items():
        to_spark(pdf, dcols).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(name)
        spark.sql(f"COMMENT ON TABLE {name} IS '{comment.replace(chr(39), '')}'")
        print(f"wrote {CATALOG}.{SCHEMA}.{name}")

    def run_sql(q):
        return spark.sql(q)

    def fetch(q):
        return spark.sql(q).toPandas()
else:
    import duckdb

    con = duckdb.connect()
    for name, (pdf, dcols, _) in TABLES.items():
        con.register("tmp_df", pdf)
        sel = ", ".join(f'CAST("{c}" AS DATE) AS "{c}"' if c in dcols else f'"{c}"' for c in pdf.columns)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT {sel} FROM tmp_df")
        con.unregister("tmp_df")

    def run_sql(q):
        return con.execute(q)

    def fetch(q):
        return con.execute(q).df()

# COMMAND ----------

# MAGIC %md ## 10. Gold layer: executive KPI tables
# MAGIC
# MAGIC Every gold table keeps the numerators and denominators next to the percentages, so the app can re-aggregate across any date range or group of sites without averaging averages.

# COMMAND ----------

GOLD_SQL = [
    # ---------- Site x day ----------
    """
    CREATE OR REPLACE TABLE gold_site_daily_kpis AS
    WITH spine AS (
      SELECT w.warehouse_id, w.warehouse_name, w.region, d.calendar_date AS kpi_date,
             d.is_weekend, d.is_holiday, d.is_peak_season
      FROM dim_warehouse w CROSS JOIN dim_date d
    ),
    rcv AS (
      SELECT warehouse_id, order_date AS kpi_date, COUNT(*) AS orders_received, SUM(units_ordered) AS units_ordered
      FROM fact_outbound_orders GROUP BY warehouse_id, order_date
    ),
    shp AS (
      SELECT warehouse_id, ship_date AS kpi_date,
             COUNT(*) AS orders_shipped,
             SUM(units_shipped) AS units_shipped,
             SUM(units_ordered) AS units_ordered_on_shipped,
             SUM(CASE WHEN is_on_time THEN 1 ELSE 0 END) AS orders_on_time,
             SUM(CASE WHEN is_in_full THEN 1 ELSE 0 END) AS orders_in_full,
             SUM(CASE WHEN is_otif THEN 1 ELSE 0 END) AS orders_otif,
             SUM(freight_cost_usd) AS freight_cost_usd,
             SUM(fulfillment_revenue_usd) AS fulfillment_revenue_usd
      FROM fact_outbound_orders WHERE ship_date IS NOT NULL
      GROUP BY warehouse_id, ship_date
    ),
    inb AS (
      SELECT warehouse_id, receipt_date AS kpi_date, COUNT(*) AS receipts,
             SUM(pallets_received) AS pallets_received, SUM(units_received) AS units_received,
             SUM(dock_to_stock_hours) AS dock_to_stock_hours_total,
             SUM(CASE WHEN asn_accurate THEN 1 ELSE 0 END) AS receipts_asn_accurate
      FROM fact_inbound_receipts GROUP BY warehouse_id, receipt_date
    ),
    lab AS (
      SELECT warehouse_id, work_date AS kpi_date,
             SUM(headcount) AS headcount, SUM(temp_headcount) AS temp_headcount,
             SUM(hours_worked) AS labor_hours, SUM(overtime_hours) AS overtime_hours,
             SUM(labor_cost_usd) AS labor_cost_usd,
             SUM(CASE WHEN labor_function = 'Picking' THEN units_processed ELSE 0 END) AS pick_units,
             SUM(CASE WHEN labor_function = 'Picking' THEN hours_worked ELSE 0 END) AS pick_hours
      FROM fact_labor_daily GROUP BY warehouse_id, work_date
    ),
    inv AS (
      SELECT warehouse_id, snapshot_date AS kpi_date,
             SUM(pallet_capacity) AS pallet_capacity, SUM(pallets_occupied) AS pallets_occupied,
             SUM(inventory_value_usd) AS inventory_value_usd, SUM(aged_90_plus_pallets) AS aged_90_plus_pallets
      FROM fact_inventory_daily GROUP BY warehouse_id, snapshot_date
    ),
    cc AS (
      SELECT warehouse_id, count_date AS kpi_date, COUNT(*) AS cycle_counts,
             SUM(CASE WHEN is_accurate THEN 1 ELSE 0 END) AS cycle_counts_accurate,
             SUM(ABS(variance_value_usd)) AS abs_variance_value_usd
      FROM fact_cycle_counts GROUP BY warehouse_id, count_date
    ),
    saf AS (
      SELECT warehouse_id, incident_date AS kpi_date, COUNT(*) AS safety_events,
             SUM(CASE WHEN is_recordable THEN 1 ELSE 0 END) AS recordable_incidents,
             SUM(CASE WHEN incident_type = 'Near Miss' THEN 1 ELSE 0 END) AS near_misses,
             SUM(lost_days) AS lost_days
      FROM fact_safety_incidents GROUP BY warehouse_id, incident_date
    ),
    dt AS (
      SELECT warehouse_id, downtime_date AS kpi_date, COUNT(*) AS downtime_events,
             SUM(CASE WHEN is_unplanned THEN downtime_minutes ELSE 0 END) AS unplanned_downtime_minutes
      FROM fact_equipment_downtime GROUP BY warehouse_id, downtime_date
    ),
    base AS (
      SELECT s.*,
        COALESCE(r.orders_received, 0) AS orders_received,
        COALESCE(r.units_ordered, 0) AS units_ordered,
        COALESCE(p.orders_shipped, 0) AS orders_shipped,
        COALESCE(p.units_shipped, 0) AS units_shipped,
        COALESCE(p.units_ordered_on_shipped, 0) AS units_ordered_on_shipped,
        COALESCE(p.orders_on_time, 0) AS orders_on_time,
        COALESCE(p.orders_in_full, 0) AS orders_in_full,
        COALESCE(p.orders_otif, 0) AS orders_otif,
        COALESCE(p.freight_cost_usd, 0) AS freight_cost_usd,
        COALESCE(p.fulfillment_revenue_usd, 0) AS fulfillment_revenue_usd,
        COALESCE(i.receipts, 0) AS receipts,
        COALESCE(i.pallets_received, 0) AS pallets_received,
        COALESCE(i.units_received, 0) AS units_received,
        COALESCE(i.dock_to_stock_hours_total, 0) AS dock_to_stock_hours_total,
        COALESCE(i.receipts_asn_accurate, 0) AS receipts_asn_accurate,
        COALESCE(l.headcount, 0) AS headcount,
        COALESCE(l.temp_headcount, 0) AS temp_headcount,
        COALESCE(l.labor_hours, 0) AS labor_hours,
        COALESCE(l.overtime_hours, 0) AS overtime_hours,
        COALESCE(l.labor_cost_usd, 0) AS labor_cost_usd,
        COALESCE(l.pick_units, 0) AS pick_units,
        COALESCE(l.pick_hours, 0) AS pick_hours,
        v.pallet_capacity, v.pallets_occupied, v.inventory_value_usd, v.aged_90_plus_pallets,
        COALESCE(c.cycle_counts, 0) AS cycle_counts,
        COALESCE(c.cycle_counts_accurate, 0) AS cycle_counts_accurate,
        COALESCE(c.abs_variance_value_usd, 0) AS abs_variance_value_usd,
        COALESCE(f.safety_events, 0) AS safety_events,
        COALESCE(f.recordable_incidents, 0) AS recordable_incidents,
        COALESCE(f.near_misses, 0) AS near_misses,
        COALESCE(f.lost_days, 0) AS lost_days,
        COALESCE(t.downtime_events, 0) AS downtime_events,
        COALESCE(t.unplanned_downtime_minutes, 0) AS unplanned_downtime_minutes
      FROM spine s
      LEFT JOIN rcv r ON s.warehouse_id = r.warehouse_id AND s.kpi_date = r.kpi_date
      LEFT JOIN shp p ON s.warehouse_id = p.warehouse_id AND s.kpi_date = p.kpi_date
      LEFT JOIN inb i ON s.warehouse_id = i.warehouse_id AND s.kpi_date = i.kpi_date
      LEFT JOIN lab l ON s.warehouse_id = l.warehouse_id AND s.kpi_date = l.kpi_date
      LEFT JOIN inv v ON s.warehouse_id = v.warehouse_id AND s.kpi_date = v.kpi_date
      LEFT JOIN cc c ON s.warehouse_id = c.warehouse_id AND s.kpi_date = c.kpi_date
      LEFT JOIN saf f ON s.warehouse_id = f.warehouse_id AND s.kpi_date = f.kpi_date
      LEFT JOIN dt t ON s.warehouse_id = t.warehouse_id AND s.kpi_date = t.kpi_date
    )
    SELECT b.*,
      SUM(orders_received) OVER (PARTITION BY warehouse_id ORDER BY kpi_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
        - SUM(orders_shipped) OVER (PARTITION BY warehouse_id ORDER BY kpi_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS open_backlog_orders,
      ROUND(100.0 * orders_otif / NULLIF(orders_shipped, 0), 2) AS otif_pct,
      ROUND(100.0 * orders_on_time / NULLIF(orders_shipped, 0), 2) AS on_time_pct,
      ROUND(100.0 * units_shipped / NULLIF(units_ordered_on_shipped, 0), 2) AS fill_rate_pct,
      ROUND(dock_to_stock_hours_total / NULLIF(receipts, 0), 2) AS avg_dock_to_stock_hours,
      ROUND(100.0 * pallets_occupied / NULLIF(pallet_capacity, 0), 2) AS utilization_pct,
      ROUND(pick_units / NULLIF(pick_hours, 0), 1) AS pick_uph,
      ROUND(100.0 * overtime_hours / NULLIF(labor_hours, 0), 2) AS overtime_pct,
      ROUND(labor_cost_usd / NULLIF(units_shipped, 0), 3) AS labor_cost_per_unit,
      ROUND(100.0 * cycle_counts_accurate / NULLIF(cycle_counts, 0), 2) AS inventory_accuracy_pct
    FROM base b
    """,
    # ---------- Network x day ----------
    """
    CREATE OR REPLACE TABLE gold_network_daily_kpis AS
    WITH n AS (
      SELECT kpi_date, MAX(CASE WHEN is_weekend THEN 1 ELSE 0 END) = 1 AS is_weekend,
             MAX(CASE WHEN is_holiday THEN 1 ELSE 0 END) = 1 AS is_holiday,
             MAX(CASE WHEN is_peak_season THEN 1 ELSE 0 END) = 1 AS is_peak_season,
             SUM(orders_received) AS orders_received, SUM(units_ordered) AS units_ordered,
             SUM(orders_shipped) AS orders_shipped, SUM(units_shipped) AS units_shipped,
             SUM(units_ordered_on_shipped) AS units_ordered_on_shipped, SUM(orders_on_time) AS orders_on_time,
             SUM(orders_in_full) AS orders_in_full, SUM(orders_otif) AS orders_otif,
             SUM(freight_cost_usd) AS freight_cost_usd, SUM(fulfillment_revenue_usd) AS fulfillment_revenue_usd,
             SUM(receipts) AS receipts, SUM(pallets_received) AS pallets_received, SUM(units_received) AS units_received,
             SUM(dock_to_stock_hours_total) AS dock_to_stock_hours_total, SUM(receipts_asn_accurate) AS receipts_asn_accurate,
             SUM(headcount) AS headcount, SUM(temp_headcount) AS temp_headcount, SUM(labor_hours) AS labor_hours,
             SUM(overtime_hours) AS overtime_hours, SUM(labor_cost_usd) AS labor_cost_usd,
             SUM(pick_units) AS pick_units, SUM(pick_hours) AS pick_hours,
             SUM(pallet_capacity) AS pallet_capacity, SUM(pallets_occupied) AS pallets_occupied,
             SUM(inventory_value_usd) AS inventory_value_usd, SUM(aged_90_plus_pallets) AS aged_90_plus_pallets,
             SUM(cycle_counts) AS cycle_counts, SUM(cycle_counts_accurate) AS cycle_counts_accurate,
             SUM(abs_variance_value_usd) AS abs_variance_value_usd, SUM(safety_events) AS safety_events,
             SUM(recordable_incidents) AS recordable_incidents, SUM(near_misses) AS near_misses, SUM(lost_days) AS lost_days,
             SUM(downtime_events) AS downtime_events, SUM(unplanned_downtime_minutes) AS unplanned_downtime_minutes,
             SUM(open_backlog_orders) AS open_backlog_orders
      FROM gold_site_daily_kpis GROUP BY kpi_date
    )
    SELECT 'NETWORK' AS warehouse_id, n.*,
      ROUND(100.0 * orders_otif / NULLIF(orders_shipped, 0), 2) AS otif_pct,
      ROUND(100.0 * orders_on_time / NULLIF(orders_shipped, 0), 2) AS on_time_pct,
      ROUND(100.0 * units_shipped / NULLIF(units_ordered_on_shipped, 0), 2) AS fill_rate_pct,
      ROUND(dock_to_stock_hours_total / NULLIF(receipts, 0), 2) AS avg_dock_to_stock_hours,
      ROUND(100.0 * pallets_occupied / NULLIF(pallet_capacity, 0), 2) AS utilization_pct,
      ROUND(pick_units / NULLIF(pick_hours, 0), 1) AS pick_uph,
      ROUND(100.0 * overtime_hours / NULLIF(labor_hours, 0), 2) AS overtime_pct,
      ROUND(labor_cost_usd / NULLIF(units_shipped, 0), 3) AS labor_cost_per_unit,
      ROUND(100.0 * cycle_counts_accurate / NULLIF(cycle_counts, 0), 2) AS inventory_accuracy_pct
    FROM n
    """,
    # ---------- Monthly base (sites + network) ----------
    """
    CREATE OR REPLACE TEMP VIEW v_monthly_kpis AS
    WITH m AS (
      SELECT warehouse_id, CAST(date_trunc('MONTH', kpi_date) AS DATE) AS month_start,
             SUM(orders_shipped) AS orders_shipped, SUM(orders_otif) AS orders_otif, SUM(orders_on_time) AS orders_on_time,
             SUM(units_shipped) AS units_shipped, SUM(units_ordered_on_shipped) AS units_ordered_on_shipped,
             SUM(dock_to_stock_hours_total) AS d2s_total, SUM(receipts) AS receipts,
             SUM(pallets_occupied) AS pallets_occupied, SUM(pallet_capacity) AS pallet_capacity,
             SUM(cycle_counts_accurate) AS cc_ok, SUM(cycle_counts) AS cc,
             SUM(pick_units) AS pick_units, SUM(pick_hours) AS pick_hours,
             SUM(overtime_hours) AS overtime_hours, SUM(labor_hours) AS labor_hours, SUM(labor_cost_usd) AS labor_cost_usd,
             SUM(recordable_incidents) AS recordables, SUM(unplanned_downtime_minutes) AS downtime_minutes
      FROM (SELECT {MCOLS} FROM gold_site_daily_kpis
            UNION ALL
            SELECT {MCOLS} FROM gold_network_daily_kpis) u
      GROUP BY warehouse_id, CAST(date_trunc('MONTH', kpi_date) AS DATE)
    )
    SELECT warehouse_id, month_start,
      100.0 * orders_otif / NULLIF(orders_shipped, 0) AS otif,
      100.0 * orders_on_time / NULLIF(orders_shipped, 0) AS on_time,
      100.0 * units_shipped / NULLIF(units_ordered_on_shipped, 0) AS fill_rate,
      d2s_total / NULLIF(receipts, 0) AS dock_to_stock,
      100.0 * pallets_occupied / NULLIF(pallet_capacity, 0) AS utilization,
      100.0 * cc_ok / NULLIF(cc, 0) AS inv_accuracy,
      pick_units / NULLIF(pick_hours, 0) AS pick_uph,
      100.0 * overtime_hours / NULLIF(labor_hours, 0) AS overtime_pct,
      labor_cost_usd / NULLIF(units_shipped, 0) AS labor_cost_per_unit,
      SUM(recordables) OVER (PARTITION BY warehouse_id ORDER BY month_start ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) * 200000.0
        / NULLIF(SUM(labor_hours) OVER (PARTITION BY warehouse_id ORDER BY month_start ROWS BETWEEN 11 PRECEDING AND CURRENT ROW), 0) AS trir,
      downtime_minutes / 60.0 AS downtime_hours
    FROM m
    """,
    # ---------- Monthly scorecard (long format with RAG status) ----------
    """
    CREATE OR REPLACE TABLE gold_monthly_scorecard AS
    WITH kpi_long AS (
      SELECT warehouse_id, month_start, 'OTIF' AS kpi_code, otif AS actual_value FROM v_monthly_kpis
      UNION ALL SELECT warehouse_id, month_start, 'ON_TIME', on_time FROM v_monthly_kpis
      UNION ALL SELECT warehouse_id, month_start, 'FILL_RATE', fill_rate FROM v_monthly_kpis
      UNION ALL SELECT warehouse_id, month_start, 'DOCK_TO_STOCK', dock_to_stock FROM v_monthly_kpis
      UNION ALL SELECT warehouse_id, month_start, 'UTILIZATION', utilization FROM v_monthly_kpis
      UNION ALL SELECT warehouse_id, month_start, 'INV_ACCURACY', inv_accuracy FROM v_monthly_kpis
      UNION ALL SELECT warehouse_id, month_start, 'PICK_UPH', pick_uph FROM v_monthly_kpis
      UNION ALL SELECT warehouse_id, month_start, 'OVERTIME_PCT', overtime_pct FROM v_monthly_kpis
      UNION ALL SELECT warehouse_id, month_start, 'LABOR_COST_PER_UNIT', labor_cost_per_unit FROM v_monthly_kpis
      UNION ALL SELECT warehouse_id, month_start, 'TRIR', trir FROM v_monthly_kpis
      UNION ALL SELECT warehouse_id, month_start, 'DOWNTIME_HOURS', downtime_hours FROM v_monthly_kpis
              WHERE warehouse_id <> 'NETWORK'
    ),
    tgt AS (
      SELECT l.warehouse_id, l.kpi_code,
             COALESCE(s.target_value, d.target_value) AS target_value,
             COALESCE(s.warning_value, d.warning_value) AS warning_value,
             d.kpi_name, d.unit, d.direction, d.display_order
      FROM (SELECT DISTINCT warehouse_id, kpi_code FROM kpi_long) l
      JOIN dim_kpi_target d ON d.kpi_code = l.kpi_code AND d.warehouse_id = 'ALL'
      LEFT JOIN dim_kpi_target s ON s.kpi_code = l.kpi_code AND s.warehouse_id = l.warehouse_id
    ),
    scored AS (
      SELECT l.*, t.kpi_name, t.unit, t.target_value, t.warning_value, t.direction, t.display_order,
             LAG(l.actual_value) OVER (PARTITION BY l.warehouse_id, l.kpi_code ORDER BY l.month_start) AS prior_month_value
      FROM kpi_long l JOIN tgt t ON l.kpi_code = t.kpi_code AND l.warehouse_id = t.warehouse_id
    )
    SELECT s.warehouse_id, COALESCE(w.warehouse_name, 'Network Total') AS warehouse_name, s.month_start,
           s.kpi_code, s.kpi_name, s.unit, s.display_order,
           ROUND(s.actual_value, 3) AS actual_value, s.target_value, s.warning_value, s.direction,
           ROUND(s.prior_month_value, 3) AS prior_month_value,
           ROUND(s.actual_value - s.prior_month_value, 3) AS change_vs_prior_month,
           CASE
             WHEN s.actual_value IS NULL THEN 'No Data'
             WHEN s.direction = 'higher_is_better' THEN
               CASE WHEN s.actual_value >= s.target_value THEN 'Green' WHEN s.actual_value >= s.warning_value THEN 'Amber' ELSE 'Red' END
             ELSE
               CASE WHEN s.actual_value <= s.target_value THEN 'Green' WHEN s.actual_value <= s.warning_value THEN 'Amber' ELSE 'Red' END
           END AS status
    FROM scored s LEFT JOIN dim_warehouse w ON s.warehouse_id = w.warehouse_id
    """,
    # ---------- Customer service by month ----------
    """
    CREATE OR REPLACE TABLE gold_customer_monthly_service AS
    WITH a AS (
      SELECT CAST(date_trunc('MONTH', ship_date) AS DATE) AS month_start, customer_id, warehouse_id,
             COUNT(*) AS orders_shipped, SUM(units_shipped) AS units_shipped, SUM(units_ordered) AS units_ordered,
             SUM(CASE WHEN is_otif THEN 1 ELSE 0 END) AS orders_otif,
             SUM(CASE WHEN is_on_time THEN 1 ELSE 0 END) AS orders_on_time,
             SUM(fulfillment_revenue_usd) AS fulfillment_revenue_usd,
             SUM(freight_cost_usd) AS freight_cost_usd
      FROM fact_outbound_orders WHERE ship_date IS NOT NULL
      GROUP BY CAST(date_trunc('MONTH', ship_date) AS DATE), customer_id, warehouse_id
    )
    SELECT a.month_start, a.customer_id, c.customer_name, c.segment, c.contract_tier, a.warehouse_id, w.warehouse_name,
           a.orders_shipped, a.units_shipped, a.units_ordered, a.orders_otif, a.orders_on_time,
           ROUND(100.0 * a.orders_otif / a.orders_shipped, 2) AS otif_pct,
           ROUND(100.0 * a.units_shipped / NULLIF(a.units_ordered, 0), 2) AS fill_rate_pct,
           c.otif_target_pct,
           CASE WHEN 100.0 * a.orders_otif / a.orders_shipped >= c.otif_target_pct THEN 'Met' ELSE 'Missed' END AS sla_status,
           ROUND(a.fulfillment_revenue_usd, 2) AS fulfillment_revenue_usd,
           ROUND(a.freight_cost_usd, 2) AS freight_cost_usd
    FROM a JOIN dim_customer c ON a.customer_id = c.customer_id
           JOIN dim_warehouse w ON a.warehouse_id = w.warehouse_id
    """,
    # ---------- Headline tiles: last 7 days vs prior 7 days ----------
    """
    CREATE OR REPLACE TABLE gold_exec_headline AS
    WITH b AS (SELECT MAX(kpi_date) AS as_of_date FROM gold_network_daily_kpis),
    u AS (
      SELECT warehouse_id, kpi_date, orders_shipped, orders_otif, units_shipped, units_ordered_on_shipped,
             labor_cost_usd, labor_hours, overtime_hours, recordable_incidents, unplanned_downtime_minutes,
             fulfillment_revenue_usd, open_backlog_orders, utilization_pct
      FROM gold_site_daily_kpis
      UNION ALL
      SELECT warehouse_id, kpi_date, orders_shipped, orders_otif, units_shipped, units_ordered_on_shipped,
             labor_cost_usd, labor_hours, overtime_hours, recordable_incidents, unplanned_downtime_minutes,
             fulfillment_revenue_usd, open_backlog_orders, utilization_pct
      FROM gold_network_daily_kpis
    ),
    tagged AS (
      SELECT u.*, b.as_of_date,
             CASE WHEN u.kpi_date > b.as_of_date - INTERVAL 7 DAYS THEN 1 ELSE 0 END AS is_cur
      FROM u CROSS JOIN b
      WHERE u.kpi_date > b.as_of_date - INTERVAL 14 DAYS
    ),
    agg AS (
      SELECT warehouse_id, MAX(as_of_date) AS as_of_date,
        SUM(CASE WHEN is_cur = 1 THEN orders_shipped ELSE 0 END) AS cur_orders,
        SUM(CASE WHEN is_cur = 0 THEN orders_shipped ELSE 0 END) AS prior_orders,
        SUM(CASE WHEN is_cur = 1 THEN orders_otif ELSE 0 END) AS cur_otif,
        SUM(CASE WHEN is_cur = 0 THEN orders_otif ELSE 0 END) AS prior_otif,
        SUM(CASE WHEN is_cur = 1 THEN units_shipped ELSE 0 END) AS cur_units,
        SUM(CASE WHEN is_cur = 0 THEN units_shipped ELSE 0 END) AS prior_units,
        SUM(CASE WHEN is_cur = 1 THEN units_ordered_on_shipped ELSE 0 END) AS cur_units_ord,
        SUM(CASE WHEN is_cur = 0 THEN units_ordered_on_shipped ELSE 0 END) AS prior_units_ord,
        SUM(CASE WHEN is_cur = 1 THEN labor_cost_usd ELSE 0 END) AS cur_labor_cost,
        SUM(CASE WHEN is_cur = 0 THEN labor_cost_usd ELSE 0 END) AS prior_labor_cost,
        SUM(CASE WHEN is_cur = 1 THEN overtime_hours ELSE 0 END) AS cur_ot,
        SUM(CASE WHEN is_cur = 0 THEN overtime_hours ELSE 0 END) AS prior_ot,
        SUM(CASE WHEN is_cur = 1 THEN labor_hours ELSE 0 END) AS cur_hours,
        SUM(CASE WHEN is_cur = 0 THEN labor_hours ELSE 0 END) AS prior_hours,
        SUM(CASE WHEN is_cur = 1 THEN recordable_incidents ELSE 0 END) AS cur_recordables,
        SUM(CASE WHEN is_cur = 1 THEN unplanned_downtime_minutes ELSE 0 END) / 60.0 AS cur_downtime_hours,
        SUM(CASE WHEN is_cur = 1 THEN fulfillment_revenue_usd ELSE 0 END) AS cur_revenue,
        SUM(CASE WHEN is_cur = 0 THEN fulfillment_revenue_usd ELSE 0 END) AS prior_revenue,
        MAX(CASE WHEN kpi_date = as_of_date THEN open_backlog_orders END) AS open_backlog_orders,
        MAX(CASE WHEN kpi_date = as_of_date THEN utilization_pct END) AS utilization_pct
      FROM tagged GROUP BY warehouse_id
    )
    SELECT a.warehouse_id, COALESCE(w.warehouse_name, 'Network Total') AS warehouse_name, w.latitude, w.longitude, a.as_of_date,
      a.cur_orders AS orders_shipped_7d, a.prior_orders AS orders_shipped_prior_7d,
      ROUND(100.0 * (a.cur_orders - a.prior_orders) / NULLIF(a.prior_orders, 0), 1) AS orders_change_pct,
      a.cur_units AS units_shipped_7d,
      ROUND(100.0 * a.cur_otif / NULLIF(a.cur_orders, 0), 2) AS otif_pct_7d,
      ROUND(100.0 * a.prior_otif / NULLIF(a.prior_orders, 0), 2) AS otif_pct_prior_7d,
      ROUND(100.0 * a.cur_units / NULLIF(a.cur_units_ord, 0), 2) AS fill_rate_pct_7d,
      ROUND(a.cur_labor_cost / NULLIF(a.cur_units, 0), 3) AS labor_cost_per_unit_7d,
      ROUND(a.prior_labor_cost / NULLIF(a.prior_units, 0), 3) AS labor_cost_per_unit_prior_7d,
      ROUND(100.0 * a.cur_ot / NULLIF(a.cur_hours, 0), 2) AS overtime_pct_7d,
      ROUND(100.0 * a.prior_ot / NULLIF(a.prior_hours, 0), 2) AS overtime_pct_prior_7d,
      a.cur_recordables AS recordable_incidents_7d,
      ROUND(a.cur_downtime_hours, 1) AS unplanned_downtime_hours_7d,
      ROUND(a.cur_revenue, 2) AS fulfillment_revenue_7d,
      ROUND(100.0 * (a.cur_revenue - a.prior_revenue) / NULLIF(a.prior_revenue, 0), 1) AS revenue_change_pct,
      a.open_backlog_orders, a.utilization_pct
    FROM agg a LEFT JOIN dim_warehouse w ON a.warehouse_id = w.warehouse_id
    """,
]

MCOLS = ", ".join([
    "warehouse_id", "kpi_date", "orders_shipped", "orders_otif", "orders_on_time", "units_shipped", "units_ordered_on_shipped",
    "dock_to_stock_hours_total", "receipts", "pallets_occupied", "pallet_capacity", "cycle_counts_accurate", "cycle_counts",
    "pick_units", "pick_hours", "overtime_hours", "labor_hours", "labor_cost_usd", "recordable_incidents", "unplanned_downtime_minutes",
])
for q in GOLD_SQL:
    run_sql(q.replace("{MCOLS}", MCOLS))

if ON_DATABRICKS:
    for name, comment in {
        "gold_site_daily_kpis": "Executive KPIs by site and day with numerators and denominators for safe re-aggregation.",
        "gold_network_daily_kpis": "Executive KPIs for the whole network by day (warehouse_id = NETWORK).",
        "gold_monthly_scorecard": "Monthly KPI scorecard in long format with targets, prior month and Green/Amber/Red status.",
        "gold_customer_monthly_service": "Monthly service levels by client and site against each client OTIF target.",
        "gold_exec_headline": "Headline tiles: last 7 days vs prior 7 days per site and network, plus current backlog and utilization.",
    }.items():
        spark.sql(f"COMMENT ON TABLE {name} IS '{comment}'")

# COMMAND ----------

# MAGIC %md ## 11. Data quality checks

# COMMAND ----------

checks = {
    "orders PK unique": "SELECT COUNT(*) - COUNT(DISTINCT order_id) FROM fact_outbound_orders",
    "receipts PK unique": "SELECT COUNT(*) - COUNT(DISTINCT receipt_id) FROM fact_inbound_receipts",
    "labor grain unique": "SELECT COUNT(*) FROM (SELECT warehouse_id, work_date, shift, labor_function FROM fact_labor_daily GROUP BY 1,2,3,4 HAVING COUNT(*) > 1) x",
    "ship before order": "SELECT COUNT(*) FROM fact_outbound_orders WHERE ship_date < order_date",
    "shipped > ordered": "SELECT COUNT(*) FROM fact_outbound_orders WHERE units_shipped > units_ordered",
    "open orders with OTIF flag": "SELECT COUNT(*) FROM fact_outbound_orders WHERE ship_date IS NULL AND is_otif IS NOT NULL",
    "utilization out of range": "SELECT COUNT(*) FROM fact_inventory_daily WHERE utilization_pct < 0 OR utilization_pct > 100",
    "negative backlog days": "SELECT COUNT(*) FROM gold_site_daily_kpis WHERE open_backlog_orders < 0",
    "site-days in gold (expect 5 x 365)": "SELECT COUNT(*) - 1825 FROM gold_site_daily_kpis",
}
failed = []
for name, q in checks.items():
    v = int(fetch(q).iloc[0, 0])
    print(f"{'PASS' if v == 0 else 'FAIL'}  {name}: {v}")
    if v != 0:
        failed.append(name)
assert not failed, f"Data quality checks failed: {failed}"

# COMMAND ----------

# MAGIC %md ## 12. Quick look
# MAGIC
# MAGIC Example queries the app can use:
# MAGIC ```sql
# MAGIC -- Headline tiles
# MAGIC SELECT * FROM gold_exec_headline ORDER BY warehouse_id;
# MAGIC -- Trend line for one KPI, all sites
# MAGIC SELECT kpi_date, warehouse_id, otif_pct FROM gold_site_daily_kpis WHERE kpi_date >= current_date() - INTERVAL 90 DAYS;
# MAGIC -- Latest month scorecard heat map
# MAGIC SELECT * FROM gold_monthly_scorecard WHERE month_start = (SELECT MAX(month_start) FROM gold_monthly_scorecard);
# MAGIC -- Chart annotations
# MAGIC SELECT * FROM dim_operational_event;
# MAGIC ```

# COMMAND ----------

summary = fetch("""
  SELECT warehouse_id, warehouse_name,
         ROUND(MAX(CASE WHEN kpi_code = 'OTIF' THEN actual_value END), 2) AS otif,
         ROUND(MAX(CASE WHEN kpi_code = 'UTILIZATION' THEN actual_value END), 2) AS utilization,
         ROUND(MAX(CASE WHEN kpi_code = 'INV_ACCURACY' THEN actual_value END), 2) AS inv_accuracy,
         ROUND(MAX(CASE WHEN kpi_code = 'OVERTIME_PCT' THEN actual_value END), 2) AS overtime_pct,
         ROUND(MAX(CASE WHEN kpi_code = 'TRIR' THEN actual_value END), 2) AS trir
  FROM gold_monthly_scorecard
  WHERE month_start = (SELECT MAX(month_start) FROM gold_monthly_scorecard)
  GROUP BY warehouse_id, warehouse_name ORDER BY warehouse_id
""")
if ON_DATABRICKS:
    display(summary)
else:
    print(summary.to_string(index=False))
