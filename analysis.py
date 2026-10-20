
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/train.csv"
PLOT_DIR = "plots"

os.makedirs(PLOT_DIR, exist_ok=True)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "data/train.csv nahi mili. Kaggle dataset data folder mein rakho."
    )

df = pd.read_csv(
    r"C:\Users\a\Downloads\house-prices-advanced-regression-techniques\train.csv"
)

print("\n========== DATASET OVERVIEW ==========")
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
missing = df.isnull().sum().sort_values(ascending=False)
print(missing[missing > 0].head(15))

print("\n========== STATISTICS ==========")
print(df["SalePrice"].describe())

sns.set_theme(style="whitegrid")

# 1. Sale price distribution
plt.figure(figsize=(9, 5))
sns.histplot(df["SalePrice"], bins=30, kde=True)
plt.title("Distribution of House Sale Prices")
plt.xlabel("Sale Price (USD)")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/price_distribution.png")
plt.close()

# 2. House quality vs price
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="OverallQual", y="SalePrice")
plt.title("House Quality vs Sale Price")
plt.xlabel("Overall Quality")
plt.ylabel("Sale Price (USD)")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/quality_vs_price.png")
plt.close()

# 3. Living area vs price
plt.figure(figsize=(9, 5))
sns.scatterplot(data=df, x="GrLivArea", y="SalePrice")
plt.title("Living Area vs Sale Price")
plt.xlabel("Above-Ground Living Area (sq ft)")
plt.ylabel("Sale Price (USD)")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/area_vs_price.png")
plt.close()

print("\nAnalysis completed successfully.")
print("Graphs saved in the plots folder.")
