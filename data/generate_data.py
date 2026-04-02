import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)
n = 5000  # Number of rows

# Create dataset
data = pd.DataFrame({
    "transaction_amount": np.random.normal(100, 20, n),
    "account_age_days": np.random.normal(500, 100, n),
    "num_transactions": np.random.normal(5, 2, n),
})

# Optional: Round num_transactions to positive integers
data["num_transactions"] = np.clip(np.round(data["num_transactions"]), 1, None)

# Randomly assign ~10% fraud
data["fraud"] = np.random.choice([0, 1], size=n, p=[0.9, 0.1])

# Save dataset to CSV
data.to_csv("data/reference_data.csv", index=False)

# Print count of fraud and non-fraud
print(data["fraud"].value_counts())
print("Reference dataset created with ~10% fraud")