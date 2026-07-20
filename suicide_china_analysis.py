import pandas as pd
import numpy as np

# this dataset has demographic and administrative details, one row per recorded case
df = pd.read_csv("SuicideChina.csv")

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

print("Q1. What is the average age recorded in the dataset?")
avg_age = df["Age"].mean()
print("Answer: " + str(avg_age)) 

print("Q2. Which feature has the highest standard deviation?")
print("Standard deviation for each column:")
print(df.std(numeric_only=True))

# value_counts is a nice beginner way to see how many records fall into each category
print("Q3. How many records are there for each gender?")
gender_counts = df["Sex"].value_counts()
print(gender_counts)

print("Q4. How many records are urban vs rural?")
urban_counts = df["Urban"].value_counts()
print(urban_counts)

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
