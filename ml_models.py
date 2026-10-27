"""
Courier Delivery ML Models
----------------------------
1. Delay Prediction (Classification): Will a new order be delayed? (Yes/No)
2. Delay Duration (Regression): If delayed, how many minutes?
3. Anomaly Detection: Flag suspicious delivery records (e.g. impossibly fast
   deliveries relative to distance/traffic/weather - possible fake confirmations)

Run: python ml_models.py
Outputs:
- delay_classifier.pkl
- delay_regressor.pkl
- anomaly_detector.pkl
- ml_predictions.csv (test set predictions for review)
"""

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "courier_deliveries.csv")


def load_and_prepare():
    df = pd.read_csv(CSV_PATH, parse_dates=["pickup_time", "expected_delivery_time", "actual_delivery_time"])

    # Feature engineering
    df["pickup_hour"] = df["pickup_time"].dt.hour
    df["is_delayed"] = (df["delivery_status"] == "Delayed").astype(int)

    # Encode categorical features
    le_traffic = LabelEncoder()
    le_weather = LabelEncoder()
    df["traffic_encoded"] = le_traffic.fit_transform(df["traffic_condition"])
    df["weather_encoded"] = le_weather.fit_transform(df["weather"])

    return df, le_traffic, le_weather


FEATURES = ["distance_km", "traffic_encoded", "weather_encoded", "pickup_hour", "agent_id"]


def train_delay_classifier(df):
    # Exclude "Failed" orders from delay prediction - failure is a different
    # problem (non-delivery), not a delay severity problem.
    data = df[df["delivery_status"] != "Failed"].copy()

    X = data[FEATURES]
    y = data["is_delayed"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("\n===== DELAY PREDICTION MODEL (Classification) =====")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, preds, target_names=["On-Time", "Delayed"]))

    feat_importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("Feature Importance:")
    print(feat_importance.round(3).to_string())

    return model, acc, X_test, y_test, preds


def train_delay_regressor(df):
    # Only meaningful for orders that were actually delayed
    data = df[df["delivery_status"] == "Delayed"].copy()

    X = data[FEATURES]
    y = data["delay_minutes"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    print("\n===== DELAY DURATION MODEL (Regression) =====")
    print(f"Mean Absolute Error: {mae:.2f} minutes")

    return model


def train_anomaly_detector(df):
    # Anomaly detection based on "speed" = distance per minute taken.
    # A delivery that is way too fast for its distance/traffic/weather
    # combination is suspicious (possible fake confirmation).
    data = df[df["delivery_status"] != "Failed"].copy()
    duration_minutes = (data["actual_delivery_time"] - data["pickup_time"]).dt.total_seconds() / 60
    data["speed_kmpm"] = data["distance_km"] / duration_minutes.replace(0, 0.1)

    features = ["distance_km", "traffic_encoded", "weather_encoded", "speed_kmpm"]
    X = data[features]

    model = IsolationForest(contamination=0.03, random_state=42)
    model.fit(X)

    data["anomaly_flag"] = model.predict(X)  # -1 = anomaly, 1 = normal
    anomalies = data[data["anomaly_flag"] == -1]

    print("\n===== ANOMALY DETECTION =====")
    print(f"Flagged {len(anomalies)} suspicious deliveries out of {len(data)} ({len(anomalies)/len(data)*100:.2f}%)")
    if len(anomalies) > 0:
        print("\nSample flagged records:")
        print(anomalies[["order_id", "agent_name", "distance_km", "traffic_condition", "weather", "speed_kmpm"]].head(5).to_string(index=False))

    return model, data


def save_models(clf, reg, anomaly_model, le_traffic, le_weather):
    joblib.dump(clf, os.path.join(SCRIPT_DIR, "delay_classifier.pkl"))
    joblib.dump(reg, os.path.join(SCRIPT_DIR, "delay_regressor.pkl"))
    joblib.dump(anomaly_model, os.path.join(SCRIPT_DIR, "anomaly_detector.pkl"))
    joblib.dump(le_traffic, os.path.join(SCRIPT_DIR, "le_traffic.pkl"))
    joblib.dump(le_weather, os.path.join(SCRIPT_DIR, "le_weather.pkl"))
    print("\nModels saved: delay_classifier.pkl, delay_regressor.pkl, anomaly_detector.pkl")


def main():
    df, le_traffic, le_weather = load_and_prepare()

    clf, acc, X_test, y_test, preds = train_delay_classifier(df)
    reg = train_delay_regressor(df)
    anomaly_model, anomaly_data = train_anomaly_detector(df)

    save_models(clf, reg, anomaly_model, le_traffic, le_weather)

    # Save a sample predictions file for report/dashboard use
    results = X_test.copy()
    results["actual_is_delayed"] = y_test
    results["predicted_is_delayed"] = preds
    results.to_csv(os.path.join(SCRIPT_DIR, "ml_predictions.csv"), index=False)
    print(f"\nSample predictions saved -> ml_predictions.csv")


if __name__ == "__main__":
    main()
