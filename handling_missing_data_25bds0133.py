import numpy as np
import pandas as pd
data = np.arange(15, 30).reshape(5, 3)
dfx = pd.DataFrame(
 data,
 index=['apple', 'banana', 'kiwi', 'grapes', 'mango'],
 columns=['store1', 'store2', 'store3']
)
print(dfx, "\n")
dfx['store4'] = np.nan
dfx.loc['watermelon'] = np.arange(15, 19)
dfx.loc['oranges'] = np.nan
dfx['store5'] = np.nan
dfx.loc['apple', 'store4'] = 20.
print(dfx, "\n")
print(dfx.isnull(), "\n")
print(dfx.notnull(), "\n")
print(dfx.isnull().sum(), "\n")
print(dfx.isnull().sum().sum(), "\n")
print(dfx.count(), "\n")
print(dfx.store4[dfx.store4.notnull()], "\n")
print(dfx.store4.dropna(), "\n")
print(dfx.dropna(), "\n")
print(dfx.dropna(how='all'), "\n")
print(dfx.dropna(how='all', axis=1), "\n")
print(dfx.dropna(thresh=5, axis=1), "\n")
ar1 = np.array([100, 200, np.nan, 300])
ser1 = pd.Series(ar1)
print(ar1.mean(), ser1.mean(), "\n")
ser2 = dfx.store4
print(ser2.sum())
print(ser2.mean())
print(ser2.cumsum(), "\n")
filledDf = dfx.fillna(0)
print(filledDf, "\n")
print(dfx.mean(numeric_only=True), "\n")
print(filledDf.mean(numeric_only=True), "\n")
print(dfx.store4.ffill(), "\n")
print(dfx.store4.bfill(), "\n")
print(dfx.fillna({'store4': 0, 'store5': -1}), "\n")
print(dfx.store1.interpolate(), "\n")
print(dfx.store4.interpolate(), "\n")
print(dfx.fillna(dfx.mean(numeric_only=True)), "\n")
print(dfx.replace(np.nan, 999), "\n")
print(dfx.dropna(axis=1, thresh=len(dfx) - 3), "\n")
dfy = dfx.copy()
dfy['store6'] = [1, 2, np.nan, 4, np.nan, 6, np.nan]
print(dfy, "\n")
print(dfy.store6.fillna(dfy.store6.median()), "\n")
print(dfy.store6.fillna(dfy.store6.mode()[0]), "\n")
