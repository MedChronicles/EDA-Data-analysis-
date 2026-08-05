
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

df = pd.read_csv('fraudTrain.csv')

candidate_cols = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long']
numeric_cols = [c for c in candidate_cols if c in df.columns]

print("Numeric columns used:", numeric_cols)
print("\nOriginal data (first 5 rows):")
print(df[numeric_cols].head())

min_max_scaler = MinMaxScaler()
df_minmax = pd.DataFrame(
    min_max_scaler.fit_transform(df[numeric_cols]),
    columns=[c + '_minmax' for c in numeric_cols]
)

standard_scaler = StandardScaler()
df_standard = pd.DataFrame(
    standard_scaler.fit_transform(df[numeric_cols]),
    columns=[c + '_standard' for c in numeric_cols]
)

robust_scaler = RobustScaler()
df_robust = pd.DataFrame(
    robust_scaler.fit_transform(df[numeric_cols]),
    columns=[c + '_robust' for c in numeric_cols]
)

result = pd.concat([df[numeric_cols], df_minmax, df_standard, df_robust], axis=1)

print("\nMin-Max scaled data (first 5 rows):")
print(df_minmax.head())

print("\nStandardized data (first 5 rows):")
print(df_standard.head())

print("\nRobust scaled data (first 5 rows):")
print(df_robust.head())

result.to_csv('fraudTrain_scaled.csv', index=False)
print("\nSaved all scaled features to fraudTrain_scaled.csv")
