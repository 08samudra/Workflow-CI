import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train():
    mlflow.set_experiment("HousingPricePrediction")
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="mlflow-project-run"):
        df = pd.read_csv("MLProject/housing_price_cleaned.csv")
        X = df.drop(columns="Price")
        y = df["Price"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

if __name__ == "__main__":
    train()
mlflow_train.yml