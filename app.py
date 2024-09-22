from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
import json

app = Flask(__name__)

# Load the model and scaler
knn_model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load the dataset
merged_df_with_population = pd.read_csv("merged_df_population_hotspots.csv", low_memory=False)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if not data or "region" not in data:
        return jsonify({"error": "Invalid input"}), 400

    region_name = data["region"]

    # Filter the dataset for the given region
    region_data = merged_df_with_population[
        merged_df_with_population["Country/Region"] == region_name
    ]

    # If no data is found for the region
    if region_data.empty:
        return jsonify({"error": "Region not found"}), 404

    # Prepare data for prediction
    features = region_data[
        ["Daily_Confirmed", "Infection_Rate", "Rolling_Daily_Confirmed"]
    ]
    scaled_features = scaler.transform(features)

    # Make predictions
    predictions = knn_model.predict(scaled_features)
    region_data["Hotspot"] = predictions

    # Filter hotspots
    hotspots = region_data[region_data["Hotspot"] == True]

    # Prepare data for the map
    hotspots_list = hotspots[["Lat", "Long", "Country/Region"]].to_dict(
        orient="records"
    )

    return jsonify(hotspots_list)


if __name__ == "__main__":
    app.run(debug=True)
