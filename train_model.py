
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data/train.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("Pehle data/train.csv download karo.")

import pandas as pd

df = pd.read_csv(
    r"C:\Users\a\Downloads\house-prices-advanced-regression-techniques\train.csv"
)

print("Dataset shape:", df.shape)
print("Dataset columns:", df.columns.tolist())

FEATURES = [
    "GrLivArea",
    "OverallQual",
    "YearBuilt",
    "TotalBsmtSF",
    "GarageCars",
    "FullBath",
    "BedroomAbvGr",
    "LotArea",
]

TARGET = "SalePrice"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]),
    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeRegressor(max_depth=8, random_state=42)),
    ]),
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=150,
            random_state=42,
            n_jobs=-1
        )),
    ]),
}

results = []
best_model = None
best_name = None
best_r2 = float("-inf")

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2 Score": round(r2, 4),
    })

    print(f"\n{name}")
    print(f"MAE: ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R2 Score: {r2:.4f}")

    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_name = name

joblib.dump(
    {
        "model": best_model,
        "features": FEATURES,
        "model_name": best_name,
    },
    f"{MODEL_DIR}/price_model.pkl",
)

results_df = pd.DataFrame(results).sort_values(
    by="R2 Score", ascending=False
)
results_df.to_csv(f"{MODEL_DIR}/model_results.csv", index=False)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))
print(f"\nBest model: {best_name}")
print(f"Best R2 Score: {best_r2:.4f}")
print("Saved model: models/price_model.pkl")
