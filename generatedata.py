"""
Courier Delivery Data Generator
--------------------------------
Generates a synthetic dataset simulating real courier delivery operations.
Output: courier_deliveries.csv

Columns:
- order_id
- agent_id, agent_name
- pickup_time, expected_delivery_time, actual_delivery_time
- distance_km
- traffic_condition (Low/Medium/High)
- weather (Clear/Rain/Storm)
- delivery_status (On-Time/Delayed/Failed)
- delay_minutes
- customer_rating (1-5)
"""

import random
import csv
import os
import argparse
from datetime import datetime, timedelta

random.seed(42)

NUM_ORDERS = 2000
NUM_AGENTS = 25

AGENT_NAMES = [f"Agent_{i}" for i in range(1, NUM_AGENTS + 1)]
TRAFFIC_LEVELS = ["Low", "Medium", "High"]
WEATHER_TYPES = ["Clear", "Rain", "Storm"]

# Some agents are intentionally "slower" or "faster" to create realistic variance
agent_speed_factor = {agent: random.uniform(0.7, 1.4) for agent in AGENT_NAMES}

def random_pickup_time():
    # Simulation window = last 180 days from today, so the dataset always
    # looks "current" whenever this script is run, instead of being pinned
    # to one hardcoded year.
    start = datetime.now() - timedelta(days=180)
    random_days = random.randint(0, 180)
    random_seconds = random.randint(0, 24 * 3600)
    return start + timedelta(days=random_days, seconds=random_seconds)

def generate_order(order_id):
    agent = random.choice(AGENT_NAMES)
    agent_id = AGENT_NAMES.index(agent) + 1

    pickup_time = random_pickup_time()
    distance_km = round(random.uniform(1, 30), 2)
    traffic = random.choices(TRAFFIC_LEVELS, weights=[0.4, 0.4, 0.2])[0]
    weather = random.choices(WEATHER_TYPES, weights=[0.6, 0.3, 0.1])[0]

    # Base delivery time estimate (minutes) based on distance
    base_time = distance_km * 3  # ~3 min per km baseline

    # Traffic/weather impact
    traffic_multiplier = {"Low": 1.0, "Medium": 1.3, "High": 1.7}[traffic]
    weather_multiplier = {"Clear": 1.0, "Rain": 1.25, "Storm": 1.6}[weather]

    expected_time_minutes = base_time * 1.0  # company's promised/expected time (ideal conditions)
    actual_time_minutes = base_time * traffic_multiplier * weather_multiplier * agent_speed_factor[agent]

    # Add some random noise
    actual_time_minutes *= random.uniform(0.9, 1.15)

    expected_delivery_time = pickup_time + timedelta(minutes=expected_time_minutes)
    actual_delivery_time = pickup_time + timedelta(minutes=actual_time_minutes)

    delay_minutes = round((actual_delivery_time - expected_delivery_time).total_seconds() / 60, 1)

    # Small chance of failed delivery
    failed = random.random() < 0.03

    if failed:
        delivery_status = "Failed"
        customer_rating = random.choice([1, 2])
    elif delay_minutes > 10:
        delivery_status = "Delayed"
        customer_rating = random.choice([2, 3, 3, 4])
    else:
        delivery_status = "On-Time"
        customer_rating = random.choice([4, 5, 5])

    return {
        "order_id": order_id,
        "agent_id": agent_id,
        "agent_name": agent,
        "pickup_time": pickup_time.strftime("%Y-%m-%d %H:%M:%S"),
        "expected_delivery_time": expected_delivery_time.strftime("%Y-%m-%d %H:%M:%S"),
        "actual_delivery_time": actual_delivery_time.strftime("%Y-%m-%d %H:%M:%S"),
        "distance_km": distance_km,
        "traffic_condition": traffic,
        "weather": weather,
        "delivery_status": delivery_status,
        # Use 0 instead of None/NaN so downstream pandas/ML code never hits
        # missing-value errors. "Failed" status already flags these rows,
        # so a 0 delay here does not mislead any delay-based analysis.
        "delay_minutes": max(delay_minutes, 0) if not failed else 0,
        "customer_rating": customer_rating,
    }

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic courier delivery data")
    parser.add_argument("--orders", type=int, default=NUM_ORDERS, help="Number of orders to generate")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path (default: courier_deliveries.csv next to this script)")
    args = parser.parse_args()

    orders = [generate_order(i) for i in range(1, args.orders + 1)]

    fieldnames = list(orders[0].keys())

    # Always resolve output path relative to this script's location,
    # so running the script from a different working directory doesn't
    # silently write the CSV somewhere unexpected.
    if args.out:
        output_path = args.out
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "courier_deliveries.csv")

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(orders)

    print(f"Generated {len(orders)} orders -> {output_path}")

if __name__ == "__main__":
    main()
