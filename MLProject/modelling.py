import os
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Muat dataset
df = pd.read_csv("housing_price_cleaned.csv")

# Split data
X = df.drop(columns=["Price"])
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mulai tracking
with mlflow.start_run():
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(model, "model")

    # Simpan plot Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Price")
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png")
    mlflow.log_artifact("actual_vs_predicted.png")

    # Simpan run_id
    with open("run_id.txt", "w") as f:
        f.write(mlflow.active_run().info.run_id)
    mlflow.log_artifact("run_id.txt")

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    print("Model training and logging completed successfully.")
