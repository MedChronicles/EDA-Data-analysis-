import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer

df = pd.DataFrame(
 {
 "Age": [25, np.nan, 30, 45, 22],
 "Salary": [50000, 60000, np.nan, 80000, 45000],
 "Category": ["A", "B", "A", np.nan, "B"],
 }
)
print(df, "\n")
mean_imputer = SimpleImputer(strategy="mean")
df["Age_Mean"] = mean_imputer.fit_transform(df[["Age"]]).ravel()
median_imputer = SimpleImputer(strategy="median")
df["Salary_Median"] = median_imputer.fit_transform(df[["Salary"]]).ravel()
mode_imputer = SimpleImputer(strategy="most_frequent")
df["Category_Mode"] = mode_imputer.fit_transform(df[["Category"]]).ravel()
const_imputer = SimpleImputer(strategy="constant", fill_value=0)
df["Age_Const"] = const_imputer.fit_transform(df[["Age"]]).ravel()
num_cols = ["Age", "Salary"]
mice_imputer = IterativeImputer(max_iter=10, random_state=42)
df_mice = df[num_cols].copy()
df_mice.iloc[:, :] = mice_imputer.fit_transform(df_mice)
df["Age_MICE"] = df_mice["Age"]
df["Salary_MICE"] = df_mice[“Salary"]
knn_imputer = KNNImputer(n_neighbors=2)
df_knn = df[num_cols].copy()
df_knn.iloc[:, :] = knn_imputer.fit_transform(df_knn)
df["Age_KNN"] = df_knn["Age"]
df["Salary_KNN"] = df_knn["Salary"]
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)
print(df, "\n")import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
df = pd.DataFrame(
 {
 "Age": [25, np.nan, 30, 45, 22],
 "Salary": [50000, 60000, np.nan, 80000, 45000],
 "Category": ["A", "B", "A", np.nan, "B"],
 }
)
print(df, "\n")
mean_imputer = SimpleImputer(strategy="mean")
df["Age_Mean"] = mean_imputer.fit_transform(df[["Age"]]).ravel()
median_imputer = SimpleImputer(strategy="median")
df["Salary_Median"] = median_imputer.fit_transform(df[["Salary"]]).ravel()
mode_imputer = SimpleImputer(strategy="most_frequent")
df["Category_Mode"] = mode_imputer.fit_transform(df[["Category"]]).ravel()
const_imputer = SimpleImputer(strategy="constant", fill_value=0)
df["Age_Const"] = const_imputer.fit_transform(df[["Age"]]).ravel()
num_cols = ["Age", "Salary"]
mice_imputer = IterativeImputer(max_iter=10, random_state=42)
df_mice = df[num_cols].copy()
df_mice.iloc[:, :] = mice_imputer.fit_transform(df_mice)
df["Age_MICE"] = df_mice["Age"]
df["Salary_MICE"] = df_mice[“Salary"]
knn_imputer = KNNImputer(n_neighbors=2)
df_knn = df[num_cols].copy()
df_knn.iloc[:, :] = knn_imputer.fit_transform(df_knn)
df["Age_KNN"] = df_knn["Age"]
df["Salary_KNN"] = df_knn["Salary"]
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)
print(df, "\n")
df2 = pd.DataFrame(
 {
 "Dept": ["Sales", "Sales", "IT", "IT", "HR", "HR"],
 "Score": [80, np.nan, 70, 75, np.nan, 60],
 }
)
print(df2, "\n")
df2["Score_GroupMean"] = df2.groupby("Dept")["Score"].transform(lambda s: s.fillna(s.mean()))
print(df2, "\n")
indicator = df[["Age", "Salary"]].isnull().astype(int).add_suffix("_missing")
df = pd.concat([df, indicator], axis=1)
print(df)
df2 = pd.DataFrame(
 {
 "Dept": ["Sales", "Sales", "IT", "IT", "HR", "HR"],
 "Score": [80, np.nan, 70, 75, np.nan, 60],
 }
)
print(df2, "\n")
df2["Score_GroupMean"] = df2.groupby("Dept")["Score"].transform(lambda s: s.fillna(s.mean()))
print(df2, "\n")
indicator = df[["Age", "Salary"]].isnull().astype(int).add_suffix("_missing")
df = pd.concat([df, indicator], axis=1)
print(df)
