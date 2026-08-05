
import pandas as pd

df = pd.read_csv('fraudTrain.csv')

print("Dataset shape:", df.shape)
print("\n'amt' summary statistics:")
print(df['amt'].describe())

bins = [0, 10, 50, 100, 500, df['amt'].max()]
bin_names = ['Very Low', 'Low', 'Medium', 'High', 'Very High']

df['amt_bin'] = pd.cut(df['amt'], bins=bins, labels=bin_names)

print("\nTransaction counts per custom amount bin:")
print(pd.Series(df['amt_bin']).value_counts().sort_index())

df['amt_quartile'] = pd.qcut(
    df['amt'], q=4,
    labels=['Q1 (cheapest 25%)', 'Q2', 'Q3', 'Q4 (priciest 25%)']
)

print("\nTransaction counts per amount quartile:")
print(pd.Series(df['amt_quartile']).value_counts().sort_index())

auto_bins = pd.cut(df['amt'], bins=5, precision=2)
print("\nAuto 5-bin distribution (equal-width, computed from min/max):")
print(pd.Series(auto_bins).value_counts().sort_index())

df.to_csv('fraudTrain_binned.csv', index=False)
print("\nSaved binned dataset to fraudTrain_binned.csv")
