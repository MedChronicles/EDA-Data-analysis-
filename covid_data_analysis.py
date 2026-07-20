import pandas as pd
import numpy as np

# this covid dataset has one row per country per date, so its basically a time series file
df = pd.read_csv("covid-data.csv")

print("TASK -> LOAD THE DATASET")

print("FIRST 5 ROWS -> ")
print(df.head()) 


print("LAST 5 ROWS -> ")
print(df.tail())

shape_of_data = df.shape
print("Dataset Shape is:")
print(shape_of_data)

print("Q1. How many records are present in the dataset?")
total_rows = len(df)
print("Answer: " + str(total_rows) + " records")

print("Q2. How many features (columns) are available?")
total_cols = len(df.columns)
print("Answer: " + str(total_cols) + " columns")

print("TASK  DISPLAY METADATA")

print("Column Names:")
print(df.columns)
print("")

print("Data Types of Columns:")
print(df.dtypes)

print("Dataset Info:")
df.info() 
print("Descriptive Statistics:")
print(df.describe())

print("Q1. What is the average new cases reported per record?")
avg_cases = df["new_cases"].mean()
print("Answer: " + str(avg_cases)) 

print("Q2. Which feature has the highest standard deviation?")
print("Standard deviation for each column:")
print(df.std(numeric_only=True))

# location column tells us the country name, nunique gives count of distinct countries
print("Q3. How many unique countries/locations are in the dataset?")
unique_countries = df["location"].nunique()
print("Answer: " + str(unique_countries) + " countries")

# idxmax gives the row index where total_cases is the biggest, then we just look up that row
print("Q4. Which country recorded the single highest total cases value, and what was it?")
max_cases_row = df.loc[df["total_cases"].idxmax()]
print("Answer: " + str(max_cases_row["location"]) + " with " + str(max_cases_row["total_cases"]) + " total cases")

print("Missing values count for each column:")
missing_stuff = df.isnull().sum()
print(missing_stuff)

print("Missing values percentage:")
missing_pct = (df.isnull().sum() / len(df)) * 100
print(missing_pct)

print("Missing values sorted:")
print(missing_stuff.sort_values(ascending=False))

print("What percentage of data is missing overall?")
total_missing = df.isnull().sum().sum()
total_cells = len(df) * len(df.columns)
overall_percent = (total_missing / total_cells) * 100
print("Answer: " + str(overall_percent) + "%")
