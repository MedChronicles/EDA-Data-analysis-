
import pandas as pd
import numpy as np

df = pd.read_csv('fraudTrain.csv')

print("Dataset shape:", df.shape)

Q1 = df['amt'].quantile(0.25)
Q3 = df['amt'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"\nQ1 = {Q1:.2f}, Q3 = {Q3:.2f}, IQR = {IQR:.2f}")
print(f"IQR outlier bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")

iqr_outliers = df[(df['amt'] < lower_bound) | (df['amt'] > upper_bound)]
print(f"IQR method flagged {len(iqr_outliers)} outlier transactions out of {len(df)}")
print(iqr_outliers[['amt']].sort_values('amt', ascending=False).head(10))

mean_amt = df['amt'].mean()
std_amt = df['amt'].std()
df['amt_zscore'] = (df['amt'] - mean_amt) / std_amt

z_outliers = df[np.abs(df['amt_zscore']) > 3]
print(f"\nZ-score method (|z| > 3) flagged {len(z_outliers)} outlier transactions")
print(z_outliers[['amt', 'amt_zscore']].sort_values('amt', ascending=False).head(10))

df_clean = df[(df['amt'] >= lower_bound) & (df['amt'] <= upper_bound)].copy()
print(f"\nCleaned dataset shape: {df_clean.shape} "
      f"(removed {len(df) - len(df_clean)} rows using the IQR rule)")

iqr_outliers.to_csv('fraudTrain_outliers.csv', index=False)
df_clean.to_csv('fraudTrain_cleaned.csv', index=False)
print("\nSaved flagged outliers to fraudTrain_outliers.csv")
print("Saved cleaned (outlier-free) dataset to fraudTrain_cleaned.csv")
