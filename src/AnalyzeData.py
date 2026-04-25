import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ======================================
# 1. Load data
# ======================================
file_path = r"..\DB_tiktok\tiktok_train.csv"

df = pd.read_csv(file_path, encoding="utf-8-sig")

# ======================================
# 2. Columns to analyze
# ======================================
cols = ["likes", "views", "shares"]

# ======================================
# 3. Histogram original data
# ======================================
for col in cols:
    plt.figure(figsize=(8, 5))
    plt.hist(df[col].dropna(), bins=50)
    plt.title(f"Distribution of {col} (Original)")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(True)

    plt.savefig(f"{col}_hist_original.png")
    plt.show()

# ======================================
# 4. Boxplot
# ======================================
for col in cols:
    plt.figure(figsize=(8, 4))
    plt.boxplot(df[col].dropna())
    plt.title(f"Boxplot of {col}")
    plt.ylabel(col)
    plt.grid(True)

    plt.savefig(f"{col}_boxplot.png")
    plt.show()

# ======================================
# 5. Compare original vs log1p
# ======================================
for col in cols:
    plt.figure(figsize=(12, 5))

    # Original
    plt.subplot(1, 2, 1)
    plt.hist(df[col].dropna(), bins=50)
    plt.title(f"{col} Original")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(True)

    # Log
    plt.subplot(1, 2, 2)
    plt.hist(np.log1p(df[col].dropna()), bins=50)
    plt.title(f"log1p({col})")
    plt.xlabel(f"log1p({col})")
    plt.ylabel("Frequency")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{col}_compare_log1p.png")
    plt.show()

# ======================================
# 6. Scatter plots
# ======================================
scatter_pairs = [
    ("views", "likes"),
    ("views", "shares"),
    ("likes", "shares")
]

for x_col, y_col in scatter_pairs:
    plt.figure(figsize=(8, 6))
    plt.scatter(df[x_col], df[y_col])
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{x_col} vs {y_col}")
    plt.grid(True)

    plt.savefig(f"{x_col}_vs_{y_col}.png")
    plt.show()

# ======================================
# 7. Summary statistics
# ======================================
print("\n===== DESCRIPTIVE STATISTICS =====")
print(df[cols].describe())

print("\nAll plots saved successfully.")
