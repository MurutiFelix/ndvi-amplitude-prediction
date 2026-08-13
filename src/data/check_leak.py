import pandas as pd
df = pd.read_csv("data/processed/dl_test_residuals_dataframe.csv")
print("Unique values in True:", df['true_log_ndvi'].nunique())
print("Unique values in Pred:", df['GraphWaveNet_pred'].nunique())
print("Mean Absolute Difference:", (df['true_log_ndvi'] - df['GraphWaveNet_pred']).abs().mean())

