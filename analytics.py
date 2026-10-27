"""
Courier Delivery Analytics
---------------------------
Reads courier_deliveries.csv and computes key performance KPIs:
- Overall on-time delivery rate
- Agent-wise performance ranking
- Delay trends by traffic condition & weather
- Average delivery time
- Zone/time-based delay patterns

Run: python analytics.py
"""

import pandas as pd
import os

# Always look for the CSV next to this script, regardless of the folder
# the script is run from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "courier_deliveries.csv")


def load_data():
    df = pd.read_csv(CSV_PATH, parse_dates=["pickup_time", "expected_delivery_time", "actual_delivery_time"])
    return df


def overall_kpis(df):
    total_orders = len(df)
    on_time = (df["delivery_status"] == "On-Time").sum()
    delayed = (df["delivery_status"] == "Delayed").sum()
    failed = (df["delivery_status"] == "Failed").sum()

    on_time_rate = round((on_time / total_orders) * 100, 2)
    avg_delay = round(df.loc[df["delivery_status"] == "Delayed", "delay_minutes"].mean(), 2)
    avg_rating = round(df["customer_rating"].mean(), 2)

    print("\n===== OVERALL KPIs =====")
    print(f"Total Orders        : {total_orders}")
    print(f"On-Time             : {on_time} ({on_time_rate}%)")
    print(f"Delayed             : {delayed}")
    print(f"Failed              : {failed}")
    print(f"Avg Delay (delayed) : {avg_delay} min")
    print(f"Avg Customer Rating : {avg_rating} / 5")

    return {
        "total_orders": total_orders,
        "on_time_rate": on_time_rate,
        "delayed": int(delayed),
        "failed": int(failed),
        "avg_delay": avg_delay,
        "avg_rating": avg_rating,
    }


def agent_performance(df):
    grouped = df.groupby("agent_name").agg(
        total_orders=("order_id", "count"),
        on_time_orders=("delivery_status", lambda x: (x == "On-Time").sum()),
        avg_delay=("delay_minutes", "mean"),
        avg_rating=("customer_rating", "mean"),
    ).reset_index()

    grouped["on_time_rate_%"] = round((grouped["on_time_orders"] / grouped["total_orders"]) * 100, 2)
    grouped["avg_delay"] = grouped["avg_delay"].round(2)
    grouped["avg_rating"] = grouped["avg_rating"].round(2)

    # Composite performance score: weighted mix of on-time rate + rating - delay penalty
    grouped["performance_score"] = round(
        (grouped["on_time_rate_%"] * 0.5)
        + (grouped["avg_rating"] * 10 * 0.3)
        - (grouped["avg_delay"].fillna(0) * 0.2),
        2,
    )

    ranked = grouped.sort_values("performance_score", ascending=False).reset_index(drop=True)

    print("\n===== TOP 5 AGENTS (by performance score) =====")
    print(ranked[["agent_name", "total_orders", "on_time_rate_%", "avg_delay", "avg_rating", "performance_score"]].head(5).to_string(index=False))

    print("\n===== BOTTOM 5 AGENTS (need attention) =====")
    print(ranked[["agent_name", "total_orders", "on_time_rate_%", "avg_delay", "avg_rating", "performance_score"]].tail(5).to_string(index=False))

    return ranked


def delay_by_condition(df):
    traffic_delay = df.groupby("traffic_condition")["delay_minutes"].mean().round(2)
    weather_delay = df.groupby("weather")["delay_minutes"].mean().round(2)

    print("\n===== AVG DELAY BY TRAFFIC CONDITION =====")
    print(traffic_delay.to_string())

    print("\n===== AVG DELAY BY WEATHER =====")
    print(weather_delay.to_string())

    return traffic_delay, weather_delay


def delay_by_hour(df):
    df["pickup_hour"] = df["pickup_time"].dt.hour
    hourly_delay = df.groupby("pickup_hour")["delay_minutes"].mean().round(2)

    print("\n===== AVG DELAY BY PICKUP HOUR (top 5 worst hours) =====")
    print(hourly_delay.sort_values(ascending=False).head(5).to_string())

    return hourly_delay


def save_summary(overall, ranked, traffic_delay, weather_delay, hourly_delay):
    out_path = os.path.join(SCRIPT_DIR, "analytics_summary.csv")
    ranked.to_csv(out_path, index=False)
    print(f"\nAgent performance summary saved -> {out_path}")


def main():
    df = load_data()
    overall = overall_kpis(df)
    ranked = agent_performance(df)
    traffic_delay, weather_delay = delay_by_condition(df)
    hourly_delay = delay_by_hour(df)
    save_summary(overall, ranked, traffic_delay, weather_delay, hourly_delay)


if __name__ == "__main__":
    main()
